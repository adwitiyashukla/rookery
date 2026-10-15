import py7zr
from pyspark.sql import functions as F

from rookery import ingest


def row(**attrs):
    return "  <row " + " ".join(f'{k}="{v}"' for k, v in attrs.items()) + " />\n"


def xml(root, rows):
    return f'<?xml version="1.0" encoding="utf-8"?>\n<{root}>\n' + "".join(rows) + f"</{root}>\n"


T1, T2, T3 = "2015-01-01T10:00:00.000", "2015-01-01T11:00:00.000", "2015-01-02T10:00:00.000"

FILES = {
    "Posts.xml": xml("posts", [
        row(Id=1, PostTypeId=1, AcceptedAnswerId=3, CreationDate=T1, Score=4, Body="x", OwnerUserId=5,
            Tags="|bash|awk|", AnswerCount=2),
        row(Id=2, PostTypeId=1, CreationDate=T3, Score=1, Body="y", OwnerUserId=6, Tags="&lt;sed&gt;&lt;bash&gt;",
            AnswerCount=0),
        row(Id=3, PostTypeId=2, ParentId=1, CreationDate=T2, Score=7, Body="z", OwnerUserId=7),
        row(Id=4, PostTypeId=2, ParentId=1, CreationDate=T2, Score=0, Body="w", OwnerDisplayName="gone"),
        row(Id=5, PostTypeId=5, CreationDate=T2, Body="wiki"),
        row(Id=7, PostTypeId=1, CreationDate=T3, Score=0, Body="v", OwnerUserId=6, Tags="|vim|", AnswerCount=0),
    ]),
    "Comments.xml": xml("comments", [row(Id=1, PostId=3, Score=0, Text="t", CreationDate=T2, UserId=5)]),
    "Users.xml": xml("users", [row(Id=5, Reputation=10, CreationDate=T1, AccountId=99)]),
    "PostLinks.xml": xml("postlinks", [row(Id=1, CreationDate=T3, PostId=2, RelatedPostId=1, LinkTypeId=3)]),
    "Tags.xml": xml("tags", [row(Id=1, TagName="bash", Count=2)]),
    "Votes.xml": xml("votes", []),
    "PostHistory.xml": xml("posthistory", [
        row(Id=1, PostHistoryTypeId=2, PostId=1, CreationDate=T1, Text="body"),
        row(Id=2, PostHistoryTypeId=3, PostId=1, CreationDate=T1, Text="|bash|"),
        row(Id=3, PostHistoryTypeId=3, PostId=2, CreationDate=T3, Text="&lt;sed&gt;"),
        row(Id=4, PostHistoryTypeId=6, PostId=1, CreationDate=T2, Text="|bash|awk|"),
    ]),
}


def make_archive(tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    for name, text in FILES.items():
        (folder / name).write_text(text, encoding="utf-8")
    raw = tmp_path / "data" / "raw" / "stackexchange"
    raw.mkdir(parents=True)
    with py7zr.SevenZipFile(raw / "test.stackexchange.com.7z", "w") as zf:
        for f in sorted(folder.iterdir()):
            zf.write(f, arcname=f.name)
    return {"paths": {"data": tmp_path / "data"}, "ingest": {"keep_xml": False}}


def test_extract_and_parse(spark, tmp_path):
    cfg = make_archive(tmp_path)
    folder = ingest.extract(cfg, "test.stackexchange.com")
    assert (folder / "extracted.json").exists()
    assert not (folder / "Votes.xml").exists()
    assert not (folder / "PostHistory.xml").exists()
    history = ingest.read_xml(spark, folder / "InitialTags.xml", ingest.HISTORY)
    posts = (ingest.with_initial_tags(ingest.shape("Posts", ingest.read_table(spark, folder, "Posts")), history)
             .withColumn("stamp", F.date_format("created", "yyyy-MM-dd HH:mm:ss")).orderBy("id").collect())
    assert [r["id"] for r in posts] == [1, 2, 3, 4, 7]
    assert posts[0]["tags"] == ["bash", "awk"]
    assert posts[1]["tags"] == ["sed", "bash"]
    assert [r["initial_tags"] for r in posts] == [["bash"], ["sed"], [], [], ["vim"]]
    assert posts[3]["owner"] is None
    assert posts[2]["parent_id"] == 1
    assert posts[0]["stamp"] == "2015-01-01 10:00:00"
    links = ingest.shape("PostLinks", ingest.read_table(spark, folder, "PostLinks")).collect()
    assert (links[0]["post_id"], links[0]["related_post_id"], links[0]["link_type"]) == (2, 1, 3)


def test_sink_filters_lines_split_across_chunks(tmp_path):
    text = FILES["PostHistory.xml"].encode()
    with open(tmp_path / "out.xml", "wb") as fh:
        sink = ingest.Sink(fh, b'PostHistoryTypeId="3"')
        for start in range(0, len(text), 7):
            sink.write(text[start:start + 7])
        sink.close()
        sink.close()
    lines = (tmp_path / "out.xml").read_text().splitlines()
    assert lines[0] == "<rows>" and lines[-1] == "</rows>"
    assert [line.split()[1] for line in lines[1:-1]] == ['Id="2"', 'Id="3"']

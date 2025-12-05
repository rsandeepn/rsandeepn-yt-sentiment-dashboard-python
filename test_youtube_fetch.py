from youtube_client import extract_video_id, fetch_comments

url = "https://www.youtube.com/watch?v=0_cn6t2zrnY"
vid = extract_video_id(url)
print("Video ID:", vid)

print("Fetching Comments.....")
comments = fetch_comments(vid,max_comments=20)
print("Total comments fetched: ", len(comments))

for c in comments[:5]:
    print("_",c["text"])
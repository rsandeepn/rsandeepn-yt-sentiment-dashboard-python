from agent import analyze_comments

if __name__ == "__main__":
    # You can change this to any video URL
    url = "https://www.youtube.com/watch?v=96XB-q2-0qo"

    result = analyze_comments(url)
    print(result["summary"])

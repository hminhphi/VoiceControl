import requests
import re
import json
from bs4 import BeautifulSoup
from utils import read_config

class YouTubeSearch:
    def __init__(self, config_path = "config.yaml"):
        
        self.config_path = config_path
        self.config = read_config(path = self.config_path)['youtube']
        
        self.base_url = self.config.get("base_url", "https://www.youtube.com/results")
        self.headers = self.config.get("headers", {})

    def extract_ytinitialdata(self, html):
        """Extract and return the ytInitialData JSON from HTML source."""
        soup = BeautifulSoup(html, "html.parser")
        for script in soup.find_all("script"):
            if script.string and 'var ytInitialData = ' in script.string:
                json_text = script.string
                json_str = re.search(r'var ytInitialData = (\{.*\});', json_text, re.DOTALL)
                if json_str:
                    return json.loads(json_str.group(1))
        # Fallback: try to find with regex from raw html
        match = re.search(r'var ytInitialData = (\{.*?\});</script>', html, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError("ytInitialData not found")

    def parse_videos(self, data):
        """Parse the ytInitialData JSON for video results."""
        results = []
        try:
            sections = data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"]["sectionListRenderer"]["contents"]
            for section in sections:
                items = section.get("itemSectionRenderer", {}).get("contents", [])
                for item in items:
                    video = item.get("videoRenderer")
                    if not video:
                        continue
                    video_id = video["videoId"]
                    title = "".join([run["text"] for run in video["title"]["runs"]])
                    url = f"https://www.youtube.com/watch?v={video_id}"
                    views = video.get("viewCountText", {}).get("simpleText", "N/A")
                    published = video.get("publishedTimeText", {}).get("simpleText", "N/A")
                    channel = video.get("ownerText", {}).get("runs", [{}])[0].get("text", "N/A")
                    duration = video.get("lengthText", {}).get("simpleText", "N/A")
                    results.append({
                        "title": title,
                        "url": url,
                        "views": views,
                        "published": published,
                        "channel": channel,
                        "duration": duration
                    })
        except Exception as e:
            print("Parsing error:", e)
        return results

    def search(self, query):
        params = {"search_query": query}
        response = requests.get(self.base_url, params=params, headers=self.headers)
        if response.status_code == 200:
            html = response.text
            try:
                ytdata = self.extract_ytinitialdata(html)
                videos = self.parse_videos(ytdata)
                return videos
            except Exception as e:
                print("Error extracting data:", e)
                return []
        else:
            print("Request failed")
            return []

if __name__ == "__main__":
    config_path = "agents/infotainment/config.yaml"
    yt = YouTubeSearch(config_path=config_path)
    query = input("Enter search text: ")
    results = yt.search(query)
    for vid in results:
        print(f"Title: {vid['title']}")
        print(f"URL: {vid['url']}")
        print(f"Views: {vid['views']}")
        print(f"Published: {vid['published']}")
        print(f"Channel: {vid['channel']}")
        print(f"Duration: {vid['duration']}")
        print("-" * 40)
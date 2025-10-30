import requests

def get_web_search_tool(tavily_key):
    def web_search_tool(query):
        url = "https://api.tavily.com/search"
        try:
            resp = requests.post(url, json={"api_key": tavily_key, "query": query})
            data = resp.json()
            if "results" in data:
                return "\n".join([r["content"] for r in data["results"][:3]])
            return "No relevant web results found."
        except Exception as e:
            return f"Error calling Tavily API: {e}"
    return web_search_tool

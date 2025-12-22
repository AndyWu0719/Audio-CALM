import requests
import sys

# 配置你的 Token
API_KEY = "30deea4ea405c99c58e9dfac3d94243934ec5c26dfa510451003763f6978482b"
# 注意：这里换成你最新的 Token
DOWNLOAD_TOKEN = "dlt_f64dba7f-5087-4125-83ce-001365c6b59b" 

URL = f"https://datacollective.mozillafoundation.org/api/datasets/cmj8u3p1w0075nxxbe8bedl00/download/{DOWNLOAD_TOKEN}"

def get_real_url():
    headers = {"Authorization": f"Bearer {API_KEY}"}
    try:
        # allow_redirects=False 让我们可以捕获 302 跳转
        # stream=True 只请求头信息，不下载文件，速度极快
        response = requests.get(URL, headers=headers, allow_redirects=False, stream=True)
        
        if response.status_code in [301, 302, 303, 307, 308]:
            return response.headers['Location']
        elif response.status_code == 200:
            # 如果直接返回 200，说明没有重定向（少见），或者 Token 失效返回了网页
            print("Error: Server returned 200 OK directly. Token might be invalid or file is not on S3.", file=sys.stderr)
            return None
        else:
            print(f"Error: HTTP {response.status_code}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    url = get_real_url()
    if url:
        print(url)
    else:
        sys.exit(1)
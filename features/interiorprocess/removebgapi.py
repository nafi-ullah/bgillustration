import requests

def fetch_removebg_account_info(api_key):
    url = "https://api.remove.bg/v1.0/account"
    headers = {
        "X-Api-Key": api_key
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        account_info = response.json()
        print("Account Info:", account_info)
        return account_info
    else:
        print(f"Failed to fetch account info. Status code: {response.status_code}")
        print("Response:", response.text)
        return None

# Replace with your actual API key
api_key = ""
fetch_removebg_account_info(api_key)

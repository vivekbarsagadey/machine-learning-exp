import urllib3, requests, json

# retrieve your wml_service_credentials_username, wml_service_credentials_password, and wml_service_credentials_url from the
# Service credentials associated with your IBM Cloud Watson Machine Learning Service instanceIBM Cloud Watson Machine Learning Service instance
wml_credentials={
"url": "https://ibm-watson-ml.mybluemix.net",
"username": "3c6dec59-b1b6-4c3c-bb3f-185547a32cd7",
"password": "2f2de888-0bdb-4c45-9fed-d704e88818ad"
}

headers = urllib3.util.make_headers(basic_auth='{username}:{password}'.format(username=wml_credentials['username'], password=wml_credentials['password']))
url = '{}/v3/identity/token'.format(wml_credentials['url'])
response = requests.get(url, headers=headers)
mltoken = json.loads(response.text).get('token')

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line
payload_scoring = {"fields":["GENDER","AGE","MARITAL_STATUS","PROFESSION"],"values":[["M",27,"Single","Professional"]]}

response_scoring = requests.post('https://ibm-watson-ml.mybluemix.net/v3/wml_instances/9dda8294-a0c5-4005-bbe9-534b3af2007c/published_models/87e7109c-ec01-460c-b43a-2c46d18891ad/deployments/5e7674e1-f2bf-427f-b244-0245b4050b57/online', json=payload_scoring, headers=header)

print("response  content >>" + str(response_scoring.content))
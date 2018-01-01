import urllib3, requests, json

# retrieve your wml_service_credentials_username, wml_service_credentials_password, and wml_service_credentials_url from the
# Service credentials associated with your IBM Cloud Watson Machine Learning Service instanceIBM Cloud Watson Machine Learning Service instance
wml_credentials={
"url": "https://ibm-watson-ml.mybluemix.net",
"username": "23ead5ad-a55e-477c-9bb4-eb10c13fb8ff",
"password": "221bacea-a95d-4f3c-afe8-2f6568528172"
}

headers = urllib3.util.make_headers(basic_auth='{username}:{password}'.format(username=wml_credentials['username'], password=wml_credentials['password']))
url = '{}/v3/identity/token'.format(wml_credentials['url'])
response = requests.get(url, headers=headers)
mltoken = json.loads(response.text).get('token')

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line
payload_scoring = {"fields": ["GENDER", "AGE", "MARITAL_STATUS", "PROFESSION"], "values": [["M",27,"Single","Professional"]]}

response_scoring = requests.post('https://ibm-watson-ml.mybluemix.net/v3/wml_instances/986c9ef8-f1ac-42ae-9e87-91dd60832701/published_models/1ab9c8a6-cd25-4d73-abb6-ca861236ecfc/deployments/4ba752dc-faf2-40e5-b712-9b773377f5f4/online', json=payload_scoring, headers=header)

print("response  content >>" + str(response_scoring.content))
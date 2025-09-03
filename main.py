import base64
import requests
import json
from datetime import date
import numpy as np

from utils.archive.api_access import API_KEY, SECRET

def get_oauth2_token():
    """
    This function retrieves the OAuth2 token from the Idealista API.
    It uses the API_KEY and SECRET to authenticate and obtain the token.
    """
    # Encode the API_KEY and SECRET
    message = API_KEY + ":" + SECRET

    # Base64 encode the message
    auth = base64.b64encode(message.encode("ascii")).decode("ascii")

    # Set up headers for the request
    headers_dic = {'Authorization': 'Basic ' + auth,
                   'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
}

    # Set up parameters for the request
    params_dic = {'grant_type': 'client_credentials',
                  'scope': 'read'}

    # Make the request to get the token
    request_call = requests.post(
        "https://api.idealista.com/oauth/token",
        headers=headers_dic,
        data=params_dic
    )

    # Parse the response to get the token
    token = json.loads(request_call.text)['access_token']
    
    return token

if __name__ == "__main__":
    token = get_oauth2_token()

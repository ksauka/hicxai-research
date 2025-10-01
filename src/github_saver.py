"""
GitHubSaver utility: Save user feedback or logs directly to a GitHub repository using the GitHub API.
Requires a GitHub personal access token with repo permissions.
"""
import requests
import base64
import os

def save_to_github(repo, path, content, commit_message, github_token):
    """
    Save content to a file in a GitHub repo (creates or updates the file).
    repo: 'username/repo'
    path: path in the repo (e.g., 'feedback/user1.txt')
    content: string content to save
    commit_message: commit message
    github_token: personal access token
    """
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    # Check if file exists
    r = requests.get(api_url, headers=headers)
    if r.status_code == 200:
        sha = r.json()['sha']
    else:
        sha = None
    data = {
        "message": commit_message,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": "main"
    }
    if sha:
        data["sha"] = sha
    r = requests.put(api_url, headers=headers, json=data)
    if r.status_code in [200, 201]:
        return True
    else:
        print(f"GitHub API error: {r.status_code} {r.text}")
        return False

# Example usage in Streamlit:
# import streamlit as st
# from github_saver import save_to_github
#
# feedback = st.text_area("Your feedback")
# if st.button("Submit Feedback"):
#     success = save_to_github(
#         repo="yourusername/yourrepo",
#         path=f"feedback/{st.session_state.get('user_id','anon')}.txt",
#         content=feedback,
#         commit_message="User feedback submission",
#         github_token=st.secrets["GITHUB_TOKEN"]
#     )
#     if success:
#         st.success("Feedback saved to GitHub!")
#     else:
#         st.error("Failed to save feedback.")

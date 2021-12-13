# -*- coding: utf-8 -*-

from __future__ import division, print_function

#####################################################################################
import base64
import os
#####################################################################################################
import re
import sys
from base64 import b64decode as base64_to_text
from builtins import map, range, str, zip
#####################################################################################
from urllib.parse import parse_qs, urlparse

######################################################################################################
######################################################################################################
import github3
import pandas as pd
import requests
#####################################################################################
from bs4 import BeautifulSoup
from future import standard_library
from lxml.html import fromstring
from requests import get

import github
import wget
from attrdict import AttrDict as dict2
# noinspection PyUnresolvedReferences
from config import LOGIN, PASSWORD, Config
from github import Github
# -*- coding: utf-8 -*-
####################################################################################################
from selenium import webdriver
#######Headless PhantomJS ##############################################################################
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
# from login_data import *
from selenium.webdriver.common.keys import Keys

standard_library.install_aliases()

reload(sys)
sys.setdefaultencoding("utf8")
if sys.platform.find("win") > -1:
    print("")
    # from guidata import qthelpers  #Otherwise Error with Spyder Save

#####################################################################################################
# noinspection PyUnboundLocalVariable
DIRCWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(DIRCWD)
sys.path.append(DIRCWD + "/aapackage")

try:
    # DIRCWD = os.environ["DIRCWD"];
    from attrdict import AttrDict as dict2

    # DIRCWD = DIRCWD[ cfg["plat"]] # print(DIRCWD, flush=True)
    os.chdir(DIRCWD)
    sys.path.append(DIRCWD + "/aapackage")
    f = open(DIRCWD + "/__config/config.py")
    cfg = dict2(dict(cfg, **eval(f.read())))
    f.close()  # Load Config
    # print(cfg.github_login, flush=True)
except Exception:
    print("Project Root Directory unknown")


__path__ = DIRCWD + "/aapackage/"
__version__ = "1.0.0"
__file__ = "util.py"


config = {"login": 78, "password": 78}


# Issue with recent selenium on firefox...
# conda install -c conda-forge selenium ==2.53.6


# DesiredCapabilities.PHANTOMJS['phantomjs.page.settings.userAgent'] = 'uastring'
DesiredCapabilities.PHANTOMJS[
    "phantomjs.page.settings.userAgent"
] = "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:16.0) Gecko/20121026 Firefox/16.0"


driver = webdriver.PhantomJS(r"D:/_devs/webserver/phantomjs-1.9.8/phantomjs.exe")
driver.get("https://github.com/login")
username = driver.find_element_by_id("login_field")
password = driver.find_element_by_id("password")
username.clear()
username.send_keys(config.login)
password.clear()
password.send_keys(config.password)
driver.find_element_by_name("commit").click()


urlbase = "url= 'https://github.com/search?"
####Most Relevant
search_type = "l=Python&o=desc&s=&type=Code&utf8=%E2%9C%93"

### Recently Indexed
# search_type= "l=Python&o=desc&s=indexed&type=Code&utf8=%E2%9C%93"
page = "&p=1"
query = "&   "


url = urlbase + search_type + page + query
driver.get(url)
# html = driver.page_source

cfg = Config(file("D:/_devs/keypair/config.py"))
print(cfg.github_login)

# cfg.app1.name


url = "https://github.com/search?l=Python&q=%22import+jedi%22+++%22jedi.Script%28%22&type=Code&utf8=%E2%9C%93"
driver.get(url)
html = driver.page_source
print(html)


driver.close()  # Current window
driver.quit()  # All windows


######################### Using Firefox ##############################################################
driver = webdriver.Firefox()
driver.get("https://github.com/login")
username = driver.find_element_by_id("login_field")
password = driver.find_element_by_id("password")

username.clear()
username.send_keys(config.login)
password.clear()
password.send_keys(config.password)
driver.find_element_by_name("commit").click()


url = "https://github.com/search?l=Python&q=%22import+jedi%22+++%22jedi.Script%28%22&type=Code&utf8=%E2%9C%93"
driver.get(url)
html = driver.page_source
print(html)


#####
# blob-code blob-code-inner sg-annotated


LOGIN = ""
PASSWORD = ""


reload(sys)
sys.setdefaultencoding("utf8")

f = open(DIRCWD + "/__config/config.py")
cfg = dict2(eval(f.read()))
f.close()


# cfg.github_login


def run():
    driver = webdriver.PhantomJS(r"D:/_devs/webserver/phantomjs-1.9.8/phantomjs.exe")
    driver.get("https://github.com/login")
    login_field = driver.find_element_by_id("login_field")
    login_field.send_keys(LOGIN)
    pass_field = driver.find_element_by_id("password")
    pass_field.send_keys(PASSWORD)
    pass_field.send_keys(Keys.ENTER)
    list_of_dicts = []

    # INSERT KEYWORDS BELOW
    keywords = ["from config import", "login"]
    kw_query = ""

    for kw in keywords:
        kw_query = kw_query + "%22" + kw + "%22+"

    # PAGE NUMBERS HERE
    page_num = 1
    box_id = 0
    list_of_dicts = []
    for page in range(page_num):
        base_url = (
            "https://github.com/search?l=Python&p="
            + str(page + 1)
            + "&q="
            + kw_query
            + "&type=Code&utf8=%E2%9C%93"
        )
        driver.get(base_url)
        html = driver.page_source
        soup = BeautifulSoup(html, "lxml")

        # print("ok")
        for desc, blob in zip(
            soup.findAll("div", class_="d-inline-block col-10"),
            soup.findAll("div", class_="file-box blob-wrapper"),
        ):
            box_id = box_id + 1

            dict1 = {
                "keywords": keywords,
                "language": "Python",
                "box_id": "",
                "box_date": "",
                "box_text": "",
                "box_reponame": "",
                "box_repourl": "",
                "box_filename": "",
                "box_fileurl": "",
                "url_scrape": base_url,
                "page": str(page + 1),
            }

            urls = desc.findAll("a")
            dict1["box_repourl"] = "https://github.com" + urls[0]["href"]
            dict1["box_fileurl"] = "https://github.com" + urls[1]["href"]
            driver.get(dict1["box_fileurl"])
            driver.find_element_by_xpath('//*[@id="raw-url"]').click()

            # print("DOWNLOADING...")
            # print(driver.current_url)
            wget.download(driver.current_url)

            dict1["box_id"] = box_id
            dict1["box_reponame"] = desc.text.strip().split(" ")[0].split("/")[-1].strip("\n")
            dict1["box_filename"] = desc.text.strip().split("\n      –\n      ")[1].split("\n")[0]
            dict1["box_date"] = (
                desc.text.strip()
                .split("\n      –\n      ")[1]
                .split("\n")[3]
                .strip("Last indexed on ")
            )

            blob_code = """ """
            for k in blob.findAll("td", class_="blob-code blob-code-inner"):
                aux = k.text.rstrip()
                if len(aux) > 1:
                    blob_code = blob_code + "\n" + aux
            dict1["box_text"] = blob_code

            list_of_dicts.append(dict1)

    df = pd.DataFrame(list_of_dicts)
    print(df)


df.to_csv("source_code.csv")


print(blob_code)
print(df)


driver.quit()


# gh = github3.GitHub()
# gh.set_client_id(client_id, client_secret)


gh = github3.login(username="arita37", password="tokyoparis237.")


res = gh.search_code("requests auth github filename:.py language:python")


gh = Github(LOGIN, PASSWORD)
# print(list(gh.search_code('requests auth github filename:.py language:python')[:5]))

search_query = "requests auth github filename:.py language:python"
# print(gh.search_code(search_query).totalCount)

gh.search_code("HTTPAdapter in:file language:python" " repo:kennethreitz/requests")


for item in res.items():
    print(item)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


gh = Github(LOGIN, PASSWORD)
# print(list(gh.search_code('requests auth github filename:.py language:python')[:5]))

search_query = "requests auth github filename:.py language:python"
# print(gh.search_code(search_query).totalCount)

# The Search API has a custom rate limit. For requests using Basic Authentication, OAuth, or client ID and
# secret, you can make up to 30 requests per minute. For unauthenticated requests, the rate limit allows
# you to make up to 10 requests per minute.
#
# Если авторизован, то каждые 2 секунды можно слать запрос, иначе каждые 6
timeout = 2 if LOGIN and PASSWORD else 6

# Немного добавить на всякий
timeout += 0.5


search_result = gh.search_code(search_query)
total_count = search_result.totalCount
page = 0

data = search_result.get_page(page)
print(data[0])
print(dir(data[0]))
print(data[0].url)
print(data[0].content)
print(base64_to_text(data[0].content.encode()).decode())
print(data[0].html_url)

# get user from repo url
user = data[0].html_url.split("/")[3]
print(user)

# i = 1
# while total_count > 0:
#     data = search_result.get_page(page)
#     for result in data:
#         print(i, result)
#         i += 1
#
#     print('page: {}, total: {}, results: {}'.format(page, total_count, len(data)))
#     page += 1
#     total_count -= len(data)
#
#     # Задержка запросов, чтобы гитхаб не блокировал временно доступ
#     time.sleep(timeout)


# i = 1
# for match in gh.search_code(search_query):
#     print(i, match)
#     i += 1
#
#     time.sleep(timeout)
#
#     # print(dir(match))
#     # break


keyword = ["import jedi", "jedi.Script("]

ss = " ".join(keyword)

with open("git_" + keyword.replace("'", "_") + ".txt", "a") as f:
    base = "https://raw.githubusercontent.com"
    resp = requests.get(
        "https://github.com/search?q={}&type=Code&ref=searchresults".format(keyword)
    )
    for language in re.findall('<a href="(.*?)".*?ter-item">', resp.text):
        for page in range(1, 100):
            req = requests.get(language + "&p={}".format(page))
            for url in re.findall('a href="(.*?)" title', req.text):
                raw = base + url.replace("/blob", "")
                print(raw + "\n")
                # f.write(raw + '\n')
                # print(raw)


"""
https://github.com/search?l=Python&q=%22import+jedi%22+++%22jedi.Script%28%22&type=Code&utf8=%E2%9C%93



"""

"""

import requests

def googleSearch(query):
    with requests.session() as c:
        url = 'https://www.google.co.in'
        query = {'q': query}
        urllink = requests.get(url, params=query)
        print(urllink)

googleSearch('Linkin Park')








you may downgrade your selenium by

pip install selenium==2.53.6

This has solved my issue






# Wait for the dynamically loaded elements to show up
WebDriverWait(wd, 10).until(
    EC.visibility_of_element_located((By.CLASS_NAME, "pricerow")))


html = driver.page_source
html


driver.find_element_by_css_select

# grab results
for link in results.find_elements_by_css_selector("a.gs-title"):
    print link.get_attribute("href")






"""


raw = get("https://www.google.com/search?q=StackOverflow").text
page = fromstring(raw)

for result in pg.cssselect(".r a"):
    url = result.get("href")
    if url.startswith("/url?"):
        url = parse_qs(urlparse(url).query)["q"]
    print(url[0])


def get_repos(since=0):
    """
    Left as an exercise for OP
    """
    url = "http://api.github.com/repositories"
    data = "{{since: {}}}".format(since)
    response = requests.get(url, data=data)

    if response.status_code == 403:
        print("Problem making request! {}".format(response.status_code))
        print(response.headers)

    matches = re.match(r"<.+?>", response.headers["Link"])
    next = matches.group(0)[1:-1]

    return response.json(), next


def get_repo(url):
    """
    Left as an exercise for OP
    """
    return requests.get(url).json()


def get_readme(url):
    """
    Left as an exercise for OP
    """
    url += "/readme"
    return requests.get(url).json()


# todo: return array of all commits so we can examine each one
def get_repo_sha(url):
    """
    Left as an exercise for OP
    """
    commits = requests.get(url + "/commits").json()
    return commits[0]["sha"]


def get_file_content(item):
    """
    Left as an exercise for OP
    """
    ignore_extensions = ["jpg"]
    filename, extension = os.path.splitext(item["path"])

    if extension in ignore_extensions:
        return []

    content = requests.get(item["url"]).json()
    lines = content["content"].split("\n")
    lines = map(base64.b64decode, lines)

    print("Path: ".format(item["path"]))
    print("Lines: ".format("".join(lines[:5])))

    return "".join(lines)


def get_repo_contents(url, sha):
    """
    Left as an exercise for OP
    """
    url += "/git/trees/{}?recursive=1".format(sha)

    return requests.get(url).json()


def process_repo_contents(repo_contents):
    """
    Left as an exercise for OP
    """
    for tree in repo_contents["tree"]:
        content_type = tree["type"]
        print("content_type --- {}".format(content_type))

        if content_type == "blob":
            github.get_file_content(tree)
            print("***blob***")
        elif content_type == "tree":
            print("***tree***")


if __name__ == "__main__":
    repos, next = github.get_repos()
    for repo in repos[0:10]:
        sha = github.get_repo_sha(repo["url"])
        repo_json = github.get_repo_contents(repo["url"], sha)
        process_repo_contents(repo_json)


def getRepo(url):
    response = requests.get(url)
    return response.json()


def getReadMe(url):
    url = url + "/readme"
    response = requests.get(url)
    return response.json()


# todo: return array of all commits so we can examine each one
def getRepoSHA(url):
    # /repos/:owner/:repo/commits
    commits = requests.get(url + "/commits").json()
    return commits[0]["sha"]


def getFileContent(item):
    ignoreExtensions = ["jpg"]
    filename, extension = os.path.splitext(item["path"])
    if extension in ignoreExtensions:
        return []
    content = requests.get(item["url"]).json()
    lines = content["content"].split("\n")
    lines = map(base64.b64decode, lines)
    print("path", item["path"])
    print("lines", "".join(lines[:5]))
    return "".join(lines)


def getRepoContents(url, sha):
    # /repos/:owner/:repo/git/trees/:sha?recursive=1
    url = url + ("/git/trees/%s?recursive=1" % sha)
    # print 'url', url
    response = requests.get(url)
    return response.json()


################################################################################################
'''

http://github3py.readthedocs.io/en/master/api.html



https://developer.github.com/v3/search/



github3.search_code(query, sort=None, order=None, per_page=None, text_match=False, number=-1, etag=None)
Find code via the code search API.

Warning You will only be able to make 5 calls with this or other search functions. To raise the rate-limit on this set of endpoints, create an authenticated GitHub Session with login.
The query can contain any combination of the following supported qualifiers:

in Qualifies which fields are searched. With this qualifier you can restrict the search to just the file contents, the file path, or both.
language Searches code based on the language it’s written in.
fork Specifies that code from forked repositories should be searched. Repository forks will not be searchable unless the fork has more stars than the parent repository.
size Finds files that match a certain size (in bytes).
path Specifies the path that the resulting file must be at.
extension Matches files with a certain extension.
user or repo Limits searches to a specific user or repository.
For more information about these qualifiers, see: http://git.io/-DvAuA

Parameters:
query (str) – (required), a valid query as described above, e.g., addClass in:file language:js repo:jquery/jquery
sort (str) – (optional), how the results should be sorted; option(s): indexed; default: best match
order (str) – (optional), the direction of the sorted results, options: asc, desc; default: desc
per_page (int) – (optional)
text_match (bool) – (optional), if True, return matching search terms. See http://git.io/4ct1eQ for more information
number (int) – (optional), number of repositories to return. Default: -1, returns all available repositories
etag (str) – (optional), previous ETag header value
Returns:
generator of CodeSearchResult




import github3

from .helper import IntegrationHelper


class TestAPI(IntegrationHelper):
    def get_client(self):
        return github3.gh

    def test_emojis(self):
        """Test the ability to use the /emojis endpoint"""
        cassette_name = self.cassette_name('emojis', cls='GitHub')
        with self.recorder.use_cassette(cassette_name):
            emojis = self.gh.emojis()

        assert emojis['+1'] is not None

    def test_search_code(self):
        """Test the ability to use the code search endpoint"""
        cassette_name = self.cassette_name('search_code',
                                           cls='GitHub')
        with self.recorder.use_cassette(cassette_name):
            repos = self.gh.search_code(
                'HTTPAdapter in:file language:python'
                ' repo:kennethreitz/requests'
                )
            assert isinstance(next(repos),
                              github3.search.CodeSearchResult)

    def test_search_users(self):
        """Test the ability to use the user search endpoint"""
        cassette_name = self.cassette_name('search_users', cls='GitHub')
        with self.recorder.use_cassette(cassette_name):
            users = self.gh.search_users('tom followers:>1000')
            assert isinstance(next(users),
                              github3.search.UserSearchResult)

    def test_search_issues(self):
        """Test the ability to use the issues search endpoint"""
        cassette_name = self.cassette_name('search_issues',
                                           cls='GitHub')
        with self.recorder.use_cassette(cassette_name):
            issues = self.gh.search_issues('github3 labels:bugs')
            assert isinstance(next(issues),
                              github3.search.IssueSearchResult)

    def test_search_repositories(self):
        """Test the ability to use the repository search endpoint"""
        cassette_name = self.cassette_name('search_repositories',
                                           cls='GitHub')
        with self.recorder.use_cassette(cassette_name):
            repos = self.gh.search_repositories('github3 language:python')
            assert isinstance(next(repos),
                              github3.search.RepositorySearchResult)
    def test_search_code(self):
        """Test the ability to use the code search endpoint"""
        cassette_name = self.cassette_name('search_code',
                                           cls='GitHub')
        with self.recorder.use_cassette(cassette_name):
            repos = self.gh.search_code(
                'HTTPAdapter in:file language:python'
                ' repo:kennethreitz/requests'
                )
            assert isinstance(next(repos),
                              github3.search.CodeSearchResult)




import github3


class Search(object):

    def __init__(self):
        pass

    @staticmethod
    def repos_user():
        user = input("Enter username: ")
        usr_repos = github3.repositories_by(username=user)

        for repos in usr_repos:
            print(repos)



import github3

g = github3.github.GitHubEnterprise(url='https://github.ubc.ca',
                                    username='***',
                                    password='***',
                                    token='***')

for item in g.search.RepositorySearchResult():
    print(item)






{
  "total_count": 1,
  "incomplete_results": false,
  "items": [
    {
      "url": "https://api.github.com/repos/octocat/Spoon-Knife/commits/bb4cc8d3b2e14b3af5df699876dd4ff3acd00b7f",
      "sha": "bb4cc8d3b2e14b3af5df699876dd4ff3acd00b7f",
      "html_url": "https://github.com/octocat/Spoon-Knife/commit/bb4cc8d3b2e14b3af5df699876dd4ff3acd00b7f",
      "comments_url": "https://api.github.com/repos/octocat/Spoon-Knife/commits/bb4cc8d3b2e14b3af5df699876dd4ff3acd00b7f/comments",
      "commit": {
        "url": "https://api.github.com/repos/octocat/Spoon-Knife/git/commits/bb4cc8d3b2e14b3af5df699876dd4ff3acd00b7f",
        "author": {
          "date": "2014-02-04T14:38:36-08:00",
          "name": "The Octocat",
          "email": "octocat@nowhere.com"
        },
        "committer": {
          "date": "2014-02-12T15:18:55-08:00",
          "name": "The Octocat",
          "email": "octocat@nowhere.com"
        },
        "message": "Create styles.css and updated README",
        "tree": {
          "url": "https://api.github.com/repos/octocat/Spoon-Knife/git/trees/a639e96f9038797fba6e0469f94a4b0cc459fa68",
          "sha": "a639e96f9038797fba6e0469f94a4b0cc459fa68"
        },
        "comment_count": 8
      },




'''

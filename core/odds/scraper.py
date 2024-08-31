import json
import os
import re
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from core.logger import logger

league_name_mapping = {'serie_a': 'serie-a',
                       'premier_league': 'premier-league',
                       'premier_2': 'championship',
                       'liga': 'laliga'}


def _get_odd_event(odd_item, side_ids):
    bet_1, bet_x, bet_2 = None, None, None

    odds = odd_item['odds']
    for x in odds:
        if x['eventParticipantId'] is None:
            bet_x = x['value']
        else:
            side = side_ids.get(x['eventParticipantId'])
            if side == 'HOME':
                bet_1 = x['value']
            elif side == 'AWAY':
                bet_2 = x['value']

    return bet_1, bet_x, bet_2


def load_next_odds(league_name):
    league_name = league_name_mapping[league_name]
    scraper = BookmakersScraper(league_name)
    scraper.scrape_bookmakers()

    bookmaker_mapping = {}

    for x in scraper.bookmakers:
        bookmaker_id = x['bookmaker']['id']
        bookmaker_name = x['bookmaker']['name']
        bookmaker_mapping[bookmaker_id] = bookmaker_name

    next_matches = pd.DataFrame()
    for match, side_ids in zip(scraper.prematch_odds, scraper.side_ids):

        odds_dict = {'match_day': [],
                     'date': [],
                     'HomeTeam': [],
                     'AwayTeam': [],
                     'bookmaker': [],
                     'bet_1': [],
                     'bet_X': [],
                     'bet_2': [],
                     }
        for odd_item in match['odds']:
            bookmaker_name = bookmaker_mapping[odd_item['bookmakerId']]
            bet_1, bet_x, bet_2 = _get_odd_event(odd_item, side_ids)

            odds_dict['match_day'].append(match['match_day'])
            odds_dict['date'].append(match['date'])
            odds_dict['HomeTeam'].append(match['home_team'])
            odds_dict['AwayTeam'].append(match['visiting_team'])
            odds_dict['bookmaker'].append(bookmaker_name)
            odds_dict['bet_1'].append(bet_1)
            odds_dict['bet_X'].append(bet_x)
            odds_dict['bet_2'].append(bet_2)

        odds_df = pd.DataFrame(odds_dict)
        next_matches = pd.concat((next_matches, odds_df))

    next_matches = next_matches.reset_index(drop=True)
    next_matches['bet_1'] = next_matches['bet_1'].astype(float)
    next_matches['bet_X'] = next_matches['bet_X'].astype(float)
    next_matches['bet_2'] = next_matches['bet_2'].astype(float)
    next_matches['match_day'] = next_matches['match_day'].astype(int)

    return next_matches


class BookmakersScraper:
    # Constants for leagues
    LALIGA = "laliga"
    SERIE_A = "serie-a"
    SERIE_B = "serie-b"
    PREMIER_LEAGUE = 'premier-league'
    CHAMPIONSHIP = 'championship'
    CHAMPIONS_LEAGUE = "champions-league"
    CONFERENCE_LEAGUE = "conference-league"

    # Constants for URLs and templates
    BASE_URL_TEMPLATE = "https://www.flashscore.it/calcio/{league}"
    MATCH_URL_TEMPLATE = "https://6.ds.lsapp.eu/pq_graphql?_hash={hash}&eventId={match_id}&projectId=6&geoIpCode=IT&geoIpSubdivisionCode=IT55"
    EVENT_URL_TEMPLATE = "https://6.ds.lsapp.eu/pq_graphql?_hash={hash}&eventId={match_id}&projectId=6"

    # Keys for live and prematch odds
    KEY_LIVE_ODDS = "oles"
    KEY_PREMATCH_ODDS = "ope"
    KEY_EVENT_INFO = "dsof"

    # GraphQL query identifiers
    FIND_LIVE_ODDS_BY_ID = "findLiveOddsById"
    FIND_PREMATCH_ODDS_BY_ID = "findPrematchOddsById"
    FIND_EVENT_BY_ID = "findEventById"

    # Mapping league constants to URL segments
    LEAGUE_URL_MAP = {
        LALIGA: "spagna/laliga/",
        SERIE_A: "italia/serie-a/",
        SERIE_B: "italia/serie-b/",
        PREMIER_LEAGUE: "inghilterra/premier-league",
        CHAMPIONSHIP: "inghilterra/championship",
        CHAMPIONS_LEAGUE: "europa/champions-league/",
        CONFERENCE_LEAGUE: "europa/conference-league/",
    }

    def __init__(self, league):
        """
        Initialize the scraper for a specific league.
        """
        self.league = self.LEAGUE_URL_MAP.get(league)
        if not self.league:
            raise ValueError(f"Unsupported league: {league}")

        logger.debug(f"Scraping data for the {self.league} league.")

        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.formatted_date = datetime.now().strftime("%Y%m%d")

        self.bookmakers = None
        self.prematch_odds = []
        self.team_ids = []
        self.side_ids = []

    def fetch_page_content(self, url):
        """
        Fetches the content of the given URL.
        Returns the content or None if the request fails.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            logger.error(f"Failed to retrieve content from {url}: {e}")
            return None

    def parse_script_tag(self, content):
        """
        Extracts the script tag containing match data from the page content.
        Returns the script content or None if not found.
        """
        soup = BeautifulSoup(content, "html.parser")
        script_tag = soup.find(
            "script", string=re.compile(r'cjs\.initialFeeds\["summary-fixtures"\]')
        )
        if script_tag:
            return script_tag.string
        logger.error("Match data script tag not found.")
        return None

    def extract_match_data(self, script_content):
        """
        Extracts match IDs, home teams, and visiting teams from the script content.
        Returns lists of match IDs, home teams, and visiting teams.
        """
        data_match = re.search(
            r'cjs\.initialFeeds\["summary-fixtures"\]\s*=\s*{\s*data:\s*`([^`]+)`',
            script_content,
        )
        if not data_match:
            logger.error("Failed to extract match data from script content.")
            return [], [], []

        # Extract match details using regular expressions
        text = data_match.group(1)
        match_ids = re.findall(r"¬~AA÷(.*?)¬AD÷", text)
        home_teams = re.findall(r"¬CX÷(.*?)¬ER÷", text)
        visiting_teams = re.findall(r"¬AF÷(.*?)¬FK÷", text)
        match_days = re.findall(r"¬ER÷Giornata (.*?)¬RW÷0", text)
        timestamp_seconds = re.findall(r"¬AD÷(.*?)¬ADE÷", text)
        date = pd.to_datetime(timestamp_seconds, unit='s')

        # Return the first 10 matches
        # return match_ids[:10], home_teams[:10], visiting_teams[:10], match_days[:10]

        return match_ids, home_teams, visiting_teams, match_days, date

    def fetch_bookmaker_data(self, match_id, _hash):
        """
        Fetches bookmaker data for a specific match by match ID and hash.
        Returns an empty list if the request fails.
        """
        match_url = self.MATCH_URL_TEMPLATE.format(hash=_hash, match_id=match_id)

        try:
            response = requests.get(match_url)
            response.raise_for_status()

            # Determine the key for retrieving odds
            key = (
                self.FIND_PREMATCH_ODDS_BY_ID
                if _hash == self.KEY_PREMATCH_ODDS
                else self.FIND_LIVE_ODDS_BY_ID
            )

            data = response.json().get("data", {}).get(key, {})

            # If fetching live odds, extract the current odds
            if _hash == self.KEY_LIVE_ODDS:
                data = data.get("current", {})

            # Store bookmakers if not already stored
            if self.bookmakers is None:
                self.bookmakers = data.get("settings", {}).get("bookmakers", {})

            return data.get("odds", [])
        except requests.RequestException as e:
            logger.error(f"Failed to retrieve bookmaker data for match {match_id}: {e}")
            return []

    def fetch_event_info(self, match_id, _hash):
        """
        Fetches bookmaker data for a specific match by match ID and hash.
        Returns an empty list if the request fails.
        """
        match_url = self.EVENT_URL_TEMPLATE.format(hash=_hash, match_id=match_id)

        try:
            response = requests.get(match_url)
            response.raise_for_status()

            # Determine the key for retrieving odds
            key = (
                self.FIND_EVENT_BY_ID
            )

            data = response.json().get("data", {}).get(key, {})

            return data.get("eventParticipants", [])
        except requests.RequestException as e:
            logger.error(f"Failed to retrieve bookmaker data for match {match_id}: {e}")
            return []

    def scrape_bookmakers(self):
        """
        Main function to scrape match and bookmaker data for the specified league.
        """
        # Fetch the page content for the league
        page_url = self.BASE_URL_TEMPLATE.format(league=self.league)
        page_content = self.fetch_page_content(page_url)
        if not page_content:
            return

        # Extract the script content containing match data
        script_content = self.parse_script_tag(page_content)
        if not script_content:
            return

        # Extract match IDs, home teams, and visiting teams from the script
        match_ids, home_teams, visiting_teams, match_days, dates = self.extract_match_data(script_content)

        # Collect bookmaker data for each match
        for match_id, home, visiting, match_day, date in zip(match_ids, home_teams, visiting_teams, match_days, dates):
            odds_data = self.fetch_bookmaker_data(match_id, self.KEY_PREMATCH_ODDS)
            event_data = self.fetch_event_info(match_id, self.KEY_EVENT_INFO)
            self.team_ids.append({x['id']: x['name'] for x in event_data})
            self.side_ids.append({x['id']: x['type']['side'] for x in event_data})

            if odds_data:
                self.prematch_odds.append(
                    {
                        "match_id": match_id,
                        "match_day": match_day,
                        "date": date,
                        "home_team": home,
                        "visiting_team": visiting,
                        "odds": odds_data,
                        # "live": live_odds_data,
                    }
                )

    def save_data_to_json(self, file_name):
        """
        Saves the collected data (prematch odds or bookmakers) to a JSON file.
        Ensures that the directory structure exists before saving.
        """
        # Define the base output directory and file path
        base_directory = os.path.join(self.current_directory, self.formatted_date)

        # Extract subdirectory structure from self.league, removing trailing slashes
        league_subdirectory = os.path.normpath(self.league)

        # Define the full output directory based on the league
        directory = os.path.join(base_directory, league_subdirectory)

        # Define the full output file path
        output_file = os.path.join(directory, f"{file_name}.json")

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Select the appropriate data to save based on the file name
        data = self.prematch_odds if file_name == "prematch_odds" else self.bookmakers

        if data:
            try:
                with open(output_file, "w", encoding="utf-8") as outfile:
                    json.dump(data, outfile, indent=4, ensure_ascii=False)
                logger.info(f"Data saved to {output_file}")
            except IOError as e:
                logger.error(f"Failed to save data to {output_file}: {e}")
        else:
            logger.error(f"No data to save for {file_name}")


if __name__ == '__main__':
    result = load_next_odds('serie_a')
    result

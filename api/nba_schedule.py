"""
NBA Schedule and Game Information
Fetches live and upcoming games from NBA API
"""

from datetime import datetime, timedelta
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import leaguegamefinder
import time


def get_todays_games():
    """Get today's NBA games"""
    try:
        # Get today's scoreboard
        games_today = scoreboard.ScoreBoard()
        games = games_today.get_dict()

        parsed_games = []

        for game in games.get('scoreboard', {}).get('games', []):
            # Map numeric status to string
            status_code = game['gameStatus']
            if status_code == 1:
                status = 'SCHEDULED'
            elif status_code == 2:
                status = 'LIVE'
            elif status_code == 3:
                status = 'COMPLETED'
            else:
                status = 'SCHEDULED'

            parsed_games.append({
                'game_id': game['gameId'],
                'game_date': game['gameTimeUTC'][:10],
                'game_status': status,
                'home_team': {
                    'team_id': game['homeTeam']['teamId'],
                    'team_name': game['homeTeam']['teamName'],
                    'score': game['homeTeam'].get('score', 0)
                },
                'away_team': {
                    'team_id': game['awayTeam']['teamId'],
                    'team_name': game['awayTeam']['teamName'],
                    'score': game['awayTeam'].get('score', 0)
                }
            })

        return parsed_games

    except Exception as e:
        print(f"Error fetching today's games: {e}")
        return []


def get_upcoming_games(days=7):
    """Get today's NBA games from NBA API (live and upcoming games today)"""
    try:
        # Get today's games from NBA API
        games = get_todays_games()
        return games

    except Exception as e:
        print(f"Error fetching games: {e}")
        return []


def get_recent_games_from_api(days=7):
    """Get recent completed games from NBA API using leaguegamefinder"""
    try:
        from nba_api.stats.endpoints import leaguegamefinder
        from datetime import datetime, timedelta

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Format dates for NBA API
        date_from = start_date.strftime('%m/%d/%Y')
        date_to = end_date.strftime('%m/%d/%Y')

        print(f"Fetching games from {date_from} to {date_to}")

        # Get games from NBA API
        gamefinder = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=date_from,
            date_to_nullable=date_to,
            league_id_nullable='00'  # NBA only
        )

        games_df = gamefinder.get_data_frames()[0]

        if games_df.empty:
            print("No games found in date range")
            return []

        # Group by game_id to get both teams
        game_dict = {}

        for _, row in games_df.iterrows():
            game_id = row['GAME_ID']
            matchup = row['MATCHUP']
            is_home = 'vs.' in matchup

            # Parse team names from matchup
            if 'vs.' in matchup:
                parts = matchup.split(' vs. ')
            else:
                parts = matchup.split(' @ ')

            team_abbr = parts[0].strip()
            opp_abbr = parts[1].strip() if len(parts) > 1 else 'OPP'

            if game_id not in game_dict:
                game_dict[game_id] = {
                    'game_id': game_id,
                    'game_date': row['GAME_DATE'],
                    'game_status': 'COMPLETED',
                    'home_team': {'team_id': 0, 'team_name': '', 'score': 0},
                    'away_team': {'team_id': 0, 'team_name': '', 'score': 0}
                }

            if is_home:
                game_dict[game_id]['home_team'] = {
                    'team_id': int(row['TEAM_ID']),
                    'team_name': team_abbr,
                    'score': int(row.get('PTS', 0) or 0)
                }
                game_dict[game_id]['away_team']['team_name'] = opp_abbr
            else:
                game_dict[game_id]['away_team'] = {
                    'team_id': int(row['TEAM_ID']),
                    'team_name': team_abbr,
                    'score': int(row.get('PTS', 0) or 0)
                }
                game_dict[game_id]['home_team']['team_name'] = opp_abbr

        # Convert to list and sort by date descending
        games_list = list(game_dict.values())
        games_list.sort(key=lambda x: x['game_date'], reverse=True)

        # Limit to 30 games
        return games_list[:30]

    except Exception as e:
        print(f"Error fetching recent games from API: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_game_box_score(game_id: str):
    """Get player box score stats for a completed game from NBA API"""
    try:
        from nba_api.stats.endpoints import boxscoretraditionalv3

        box_score = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        data = box_score.get_dict()

        box_data = data.get('boxScoreTraditional', {})
        home_team = box_data.get('homeTeam', {})
        away_team = box_data.get('awayTeam', {})

        # Combine players from both teams
        all_players = []
        if home_team:
            for p in home_team.get('players', []):
                p['teamId'] = home_team.get('teamId')
                p['teamTricode'] = home_team.get('teamTricode')
                all_players.append(p)
        if away_team:
            for p in away_team.get('players', []):
                p['teamId'] = away_team.get('teamId')
                p['teamTricode'] = away_team.get('teamTricode')
                all_players.append(p)

        if not all_players:
            print(f"No players found for game {game_id}")
            return {}

        # Check if game has been played (check if any player has minutes)
        has_stats = any(p.get('statistics', {}).get('minutes', '') for p in all_players)
        if not has_stats:
            # Game hasn't been played yet
            return {}

        # Convert to dict keyed by player_id
        stats_dict = {}
        for player in all_players:
            player_id = int(player.get('personId', 0))
            stats = player.get('statistics', {})

            # Only include players who played (have minutes)
            if not stats.get('minutes'):
                continue

            stats_dict[player_id] = {
                'points': int(stats.get('points', 0) or 0),
                'rebounds': int(stats.get('reboundsTotal', 0) or 0),
                'assists': int(stats.get('assists', 0) or 0),
                'minutes': stats.get('minutes', '0:00'),
                'player_name': f"{player.get('firstName', '')} {player.get('familyName', '')}".strip(),
                'team_id': int(player.get('teamId', 0)),
                'team_abbreviation': player.get('teamTricode', '')
            }

        return stats_dict

    except Exception as e:
        print(f"Error fetching box score for game {game_id}: {e}")
        import traceback
        traceback.print_exc()
        return {}


def get_player_roster(game_id: str):
    """
    Get player roster for a specific game
    First try NBA API box score, then database, then team rosters
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from database_manager import DatabaseManager
    from nba_api.stats.endpoints import commonteamroster, boxscoretraditionalv3
    import pandas as pd

    # First, try to get players from box score (for completed/live games)
    try:
        box_score = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
        data = box_score.get_dict()

        box_data = data.get('boxScoreTraditional', {})
        home_team_data = box_data.get('homeTeam', {})
        away_team_data = box_data.get('awayTeam', {})

        if home_team_data and away_team_data:
            home_team_id = home_team_data.get('teamId')
            away_team_id = away_team_data.get('teamId')
            home_team_name = home_team_data.get('teamTricode', 'Home')
            away_team_name = away_team_data.get('teamTricode', 'Away')

            players = []

            # Get home team players
            for p in home_team_data.get('players', []):
                players.append({
                    'player_id': int(p.get('personId', 0)),
                    'player_name': f"{p.get('firstName', '')} {p.get('familyName', '')}".strip(),
                    'team': 'Home',
                    'team_id': home_team_id,
                    'opponent_team_id': away_team_id,
                    'is_home': True
                })

            # Get away team players
            for p in away_team_data.get('players', []):
                players.append({
                    'player_id': int(p.get('personId', 0)),
                    'player_name': f"{p.get('firstName', '')} {p.get('familyName', '')}".strip(),
                    'team': 'Away',
                    'team_id': away_team_id,
                    'opponent_team_id': home_team_id,
                    'is_home': False
                })

            if players:
                # Get game date from the box score data
                game_date = box_data.get('gameDate', '')
                if not game_date:
                    # Try to extract from game_id (format: 002YYXXXXX)
                    from datetime import datetime
                    game_date = datetime.now().strftime('%Y-%m-%d')

                return {
                    'game_id': game_id,
                    'game_date': game_date,
                    'home_team': home_team_name,
                    'away_team': away_team_name,
                    'players': players
                }
    except Exception as e:
        print(f"Could not get roster from box score: {e}")

    # Fallback: try database
    db = DatabaseManager()
    with db.get_connection() as conn:
        game_query = """
            SELECT DISTINCT
                player_id,
                player_name,
                team_id,
                game_date,
                matchup
            FROM player_game_logs
            WHERE game_id = ?
            ORDER BY player_name
        """

        game_players = pd.read_sql(game_query, conn, params=[game_id])

    if not game_players.empty:
        # Historical game found in database
        game_date = game_players.iloc[0]['game_date']

        # Get unique team IDs in this game
        team_ids = game_players['team_id'].unique()

        # Determine home/away teams by examining matchups
        home_team_id = None
        away_team_id = None
        home_team_name = None
        away_team_name = None

        for tid in team_ids:
            team_matchup = game_players[game_players['team_id'] == tid].iloc[0]['matchup']
            is_home = 'vs.' in team_matchup
            parts = team_matchup.replace(' vs. ', ' @ ').split(' @ ')
            team_abbr = parts[0].strip()
            opp_abbr = parts[1].strip() if len(parts) > 1 else 'OPP'

            if is_home:
                home_team_id = int(tid)
                home_team_name = team_abbr
                away_team_name = opp_abbr
            else:
                away_team_id = int(tid)
                away_team_name = team_abbr
                home_team_name = opp_abbr

        # Build player list with correct team assignments
        players = []
        for _, player in game_players.iterrows():
            player_team_id = int(player['team_id'])
            is_home = 'vs.' in player['matchup']

            if is_home:
                opponent_id = away_team_id if away_team_id else player_team_id + 1
            else:
                opponent_id = home_team_id if home_team_id else player_team_id + 1

            players.append({
                'player_id': int(player['player_id']),
                'player_name': player['player_name'],
                'team': 'Home' if is_home else 'Away',
                'team_id': player_team_id,
                'opponent_team_id': opponent_id,
                'is_home': is_home
            })

        return {
            'game_id': game_id,
            'game_date': game_date,
            'home_team': home_team_name or 'Home',
            'away_team': away_team_name or 'Away',
            'players': players
        }

    # Game not in database - try to get from NBA API
    try:
        # Get game details from scoreboard
        games_today = scoreboard.ScoreBoard()
        games = games_today.get_dict()

        target_game = None
        for game in games.get('scoreboard', {}).get('games', []):
            if game['gameId'] == game_id:
                target_game = game
                break

        if not target_game:
            # Game not found
            return {
                'game_id': game_id,
                'game_date': '',
                'home_team': 'Unknown',
                'away_team': 'Unknown',
                'players': []
            }

        # Get team rosters
        home_team_id = target_game['homeTeam']['teamId']
        away_team_id = target_game['awayTeam']['teamId']
        home_team_name = target_game['homeTeam']['teamName']
        away_team_name = target_game['awayTeam']['teamName']
        game_date = target_game['gameTimeUTC'][:10]

        players = []

        # Get home team roster
        try:
            home_roster = commonteamroster.CommonTeamRoster(team_id=home_team_id, season='2025-26')
            home_df = home_roster.get_data_frames()[0]

            for _, player in home_df.iterrows():
                players.append({
                    'player_id': int(player['PLAYER_ID']),
                    'player_name': player['PLAYER'],
                    'team': 'Home',
                    'team_id': home_team_id,
                    'opponent_team_id': away_team_id,
                    'is_home': True
                })

            time.sleep(0.6)  # Rate limiting
        except Exception as e:
            print(f"Error fetching home roster: {e}")

        # Get away team roster
        try:
            away_roster = commonteamroster.CommonTeamRoster(team_id=away_team_id, season='2025-26')
            away_df = away_roster.get_data_frames()[0]

            for _, player in away_df.iterrows():
                players.append({
                    'player_id': int(player['PLAYER_ID']),
                    'player_name': player['PLAYER'],
                    'team': 'Away',
                    'team_id': away_team_id,
                    'opponent_team_id': home_team_id,
                    'is_home': False
                })

            time.sleep(0.6)  # Rate limiting
        except Exception as e:
            print(f"Error fetching away roster: {e}")

        return {
            'game_id': game_id,
            'game_date': game_date,
            'home_team': home_team_name,
            'away_team': away_team_name,
            'players': players
        }

    except Exception as e:
        print(f"Error fetching game roster: {e}")
        return {
            'game_id': game_id,
            'game_date': '',
            'home_team': 'Unknown',
            'away_team': 'Unknown',
            'players': []
        }

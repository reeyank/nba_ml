"""
Data Collection Script
Fetches NBA game data from nba_api and stores in database
"""

import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from database_manager import DatabaseManager
from config import SEASONS


def collect_game_logs(seasons=None):
    """
    Collect player game logs from NBA API

    Args:
        seasons: List of seasons to collect (e.g., ['2022-23', '2023-24'])
    """
    if seasons is None:
        seasons = SEASONS

    db = DatabaseManager()

    print("=" * 60)
    print("NBA DATA COLLECTION")
    print("=" * 60)
    print(f"\nCollecting data for seasons: {seasons}")
    print()

    all_games = []

    for season in seasons:
        print(f"Fetching {season} season data...")

        try:
            # Fetch game logs from NBA API
            gamelog = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star='Regular Season',
                player_or_team_abbreviation='P'  # P for Player
            )

            # Get dataframe
            df = gamelog.get_data_frames()[0]

            print(f"  ✓ Retrieved {len(df):,} game records")

            # Rename columns to match our schema
            df = df.rename(columns={
                'PLAYER_ID': 'player_id',
                'PLAYER_NAME': 'player_name',
                'TEAM_ID': 'team_id',
                'GAME_ID': 'game_id',
                'GAME_DATE': 'game_date',
                'MATCHUP': 'matchup',
                'PTS': 'points',
                'REB': 'rebounds',
                'AST': 'assists',
                'STL': 'steals',
                'BLK': 'blocks',
                'TOV': 'turnovers',
                'MIN': 'minutes',
                'FG_PCT': 'field_goal_pct',
                'FG3_PCT': 'three_point_pct',
                'FT_PCT': 'free_throw_pct'
            })

            # Add season column
            df['season'] = season

            # Select only columns we need
            columns_to_keep = [
                'player_id', 'player_name', 'team_id', 'game_id', 'game_date',
                'season', 'matchup', 'points', 'rebounds', 'assists', 'steals',
                'blocks', 'turnovers', 'minutes', 'field_goal_pct',
                'three_point_pct', 'free_throw_pct'
            ]

            df = df[columns_to_keep]

            all_games.append(df)

            # Rate limiting - be nice to NBA API
            time.sleep(1)

        except Exception as e:
            print(f"  ✗ Error fetching {season}: {e}")
            continue

    if not all_games:
        print("\n✗ No data collected")
        return 0

    # Combine all seasons
    print(f"\nCombining data from all seasons...")
    combined_df = pd.concat(all_games, ignore_index=True)

    print(f"  ✓ Total games: {len(combined_df):,}")
    print(f"  ✓ Unique players: {combined_df['player_id'].nunique():,}")

    # Store in database
    print(f"\nStoring data in database...")

    with db.get_connection() as conn:
        combined_df.to_sql('player_game_logs', conn, if_exists='replace', index=False)
        conn.commit()

    print(f"  ✓ Data stored successfully")

    print("\n" + "=" * 60)
    print(f"DATA COLLECTION COMPLETE")
    print(f"Total records: {len(combined_df):,}")
    print("=" * 60)

    return len(combined_df)


def collect_opponent_defensive_stats(seasons=None):
    """
    Collect opponent defensive statistics
    This is a simplified version - just creates placeholder data
    """
    if seasons is None:
        seasons = SEASONS

    db = DatabaseManager()

    print("\nCollecting opponent defensive stats...")

    # Get all unique team IDs from game logs
    with db.get_connection() as conn:
        teams_df = pd.read_sql("SELECT DISTINCT team_id FROM player_game_logs", conn)

    # Create defensive stats for each team/season
    defensive_stats = []

    for season in seasons:
        for team_id in teams_df['team_id']:
            defensive_stats.append({
                'team_id': team_id,
                'season': season,
                'defensive_rating': 110.0  # Placeholder - would need advanced stats API
            })

    df = pd.DataFrame(defensive_stats)

    with db.get_connection() as conn:
        df.to_sql('opponent_defensive_stats', conn, if_exists='replace', index=False)
        conn.commit()

    print(f"  ✓ Created {len(df)} defensive stat records")


if __name__ == "__main__":
    # Collect game logs
    total_games = collect_game_logs()

    # Collect defensive stats
    if total_games > 0:
        collect_opponent_defensive_stats()

    print("\n✓ All data collection complete!")

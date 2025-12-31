"""
Simple database manager for NBA predictions
"""
import sqlite3
import pandas as pd
from contextlib import contextmanager
from config import DATABASE_PATH


class DatabaseManager:
    """Simple database manager for querying NBA data"""

    def __init__(self, db_path=None):
        self.db_path = db_path or DATABASE_PATH

    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def get_all_player_game_logs(self, seasons=None):
        """Get all player game logs for specified seasons"""
        if seasons is None:
            query = "SELECT * FROM player_game_logs ORDER BY game_date"
        else:
            placeholders = ','.join('?' * len(seasons))
            query = f"SELECT * FROM player_game_logs WHERE season IN ({placeholders}) ORDER BY game_date"

        with self.get_connection() as conn:
            if seasons:
                return pd.read_sql(query, conn, params=seasons)
            else:
                return pd.read_sql(query, conn)

    def get_player_games_before_date(self, player_id, before_date, limit=15):
        """Get player's games before a specific date (most recent first)"""
        query = """
            SELECT * FROM player_game_logs
            WHERE player_id = ? AND game_date < ?
            ORDER BY game_date DESC
            LIMIT ?
        """
        with self.get_connection() as conn:
            return pd.read_sql(query, conn, params=[player_id, before_date, limit])

    def get_team_stats_for_game(self, team_id, season):
        """Get team stats for a season"""
        try:
            query = """
                SELECT pace, offensive_rating, defensive_rating
                FROM team_game_stats
                WHERE team_id = ? AND season = ?
                LIMIT 1
            """
            with self.get_connection() as conn:
                result = pd.read_sql(query, conn, params=[team_id, season])
                if not result.empty:
                    return result.iloc[0].to_dict()
        except:
            pass
        return None

    def get_opponent_defensive_rating(self, team_id, season, game_date=None):
        """Get opponent's defensive rating"""
        try:
            query = """
                SELECT defensive_rating
                FROM opponent_defensive_stats
                WHERE team_id = ? AND season = ?
                LIMIT 1
            """
            with self.get_connection() as conn:
                result = pd.read_sql(query, conn, params=[team_id, season])
                if not result.empty:
                    return result['defensive_rating'].iloc[0]
        except:
            pass
        return None

    def get_all_features(self, seasons=None):
        """Get all features from feature store"""
        if seasons is None:
            query = "SELECT * FROM feature_store ORDER BY game_date"
        else:
            placeholders = ','.join('?' * len(seasons))
            query = f"SELECT * FROM feature_store WHERE season IN ({placeholders}) ORDER BY game_date"

        with self.get_connection() as conn:
            if seasons:
                return pd.read_sql(query, conn, params=seasons)
            else:
                return pd.read_sql(query, conn)

    def insert_features(self, features_df):
        """Insert features into feature store"""
        with self.get_connection() as conn:
            features_df.to_sql('feature_store', conn, if_exists='append', index=False)
            conn.commit()
        return True

    def get_active_model(self):
        """Get active model metadata"""
        query = "SELECT * FROM model_metadata WHERE is_active = 1 LIMIT 1"
        with self.get_connection() as conn:
            result = pd.read_sql(query, conn)
            if not result.empty:
                return result.iloc[0].to_dict()
            return None

    def insert_model_metadata(self, metadata):
        """Insert model metadata"""
        with self.get_connection() as conn:
            df = pd.DataFrame([metadata])
            df.to_sql('model_metadata', conn, if_exists='append', index=False)
            conn.commit()

    def set_active_model(self, model_version):
        """Set a model as active"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Deactivate all models
            cursor.execute("UPDATE model_metadata SET is_active = 0")
            # Activate specific model
            cursor.execute("UPDATE model_metadata SET is_active = 1 WHERE model_version = ?", [model_version])
            conn.commit()

    def get_database_stats(self):
        """Get database statistics"""
        stats = {}
        tables = ['player_game_logs', 'team_game_stats', 'opponent_defensive_stats', 'feature_store']

        with self.get_connection() as conn:
            cursor = conn.cursor()
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = cursor.fetchone()[0]
                except:
                    stats[table] = 0

        return stats

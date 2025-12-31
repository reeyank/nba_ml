#!/usr/bin/env python3
"""
Database Schema Setup
Creates all required tables for NBA predictions
"""

import sqlite3
from pathlib import Path
from config import DATABASE_PATH


def create_tables():
    """Create all database tables"""

    # Ensure directory exists
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    print("Creating database tables...")

    # Player game logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_game_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            player_name TEXT,
            team_id INTEGER,
            game_id TEXT,
            game_date TEXT NOT NULL,
            season TEXT,
            matchup TEXT,
            points INTEGER,
            rebounds INTEGER,
            assists INTEGER,
            steals INTEGER,
            blocks INTEGER,
            turnovers INTEGER,
            minutes REAL,
            field_goal_pct REAL,
            three_point_pct REAL,
            free_throw_pct REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create index on player_id and game_date for fast lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_player_date
        ON player_game_logs(player_id, game_date)
    """)

    # Team stats table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS team_game_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id INTEGER NOT NULL,
            season TEXT NOT NULL,
            pace REAL,
            offensive_rating REAL,
            defensive_rating REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Opponent defensive stats table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS opponent_defensive_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id INTEGER NOT NULL,
            season TEXT NOT NULL,
            defensive_rating REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Feature store table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_store (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            game_id TEXT,
            game_date TEXT NOT NULL,
            season TEXT,

            -- Rolling averages (5 games)
            points_avg_5 REAL,
            rebounds_avg_5 REAL,
            assists_avg_5 REAL,
            minutes_avg_5 REAL,
            field_goal_pct_avg_5 REAL,
            three_point_pct_avg_5 REAL,
            free_throw_pct_avg_5 REAL,

            -- Rolling averages (10 games)
            points_avg_10 REAL,
            rebounds_avg_10 REAL,
            assists_avg_10 REAL,
            minutes_avg_10 REAL,
            field_goal_pct_avg_10 REAL,

            -- Rolling averages (15 games)
            points_avg_15 REAL,
            rebounds_avg_15 REAL,
            assists_avg_15 REAL,
            minutes_avg_15 REAL,

            -- Trend features
            points_trend_5 REAL,
            rebounds_trend_5 REAL,
            assists_trend_5 REAL,
            minutes_trend_5 REAL,

            -- Contextual features
            is_home INTEGER,
            rest_days INTEGER,
            is_back_to_back INTEGER,

            -- Team features
            team_pace REAL,
            team_offensive_rating REAL,
            team_defensive_rating REAL,

            -- Opponent features
            opponent_team_id INTEGER,
            opponent_defensive_rating REAL,
            opponent_pace REAL,

            -- Target variables
            target_points INTEGER,
            target_rebounds INTEGER,
            target_assists INTEGER,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Model metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            model_type TEXT,
            training_date TEXT,
            model_path TEXT,
            is_active INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

    print("✓ Database tables created successfully")
    print(f"✓ Database location: {DATABASE_PATH}")


if __name__ == "__main__":
    create_tables()

#!/usr/bin/env python3
"""
Simple script to predict player stats for next game

Usage:
    python predict_next_game.py

Then follow the prompts, or edit the script to set defaults.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from prediction.predictor import GamePredictor


def main():
    """Interactive prediction script"""
    print("="*60)
    print("NBA NEXT GAME STAT PREDICTOR")
    print("="*60)
    print()

    # Initialize predictor
    try:
        predictor = GamePredictor()
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    print("\nAvailable players:")
    print("  - LeBron James")
    print("  - Stephen Curry")
    print("  - Giannis Antetokounmpo")
    print("  - Kevin Durant")
    print("  - Luka Doncic")
    print("  - Nikola Jokic")
    print("  - Joel Embiid")
    print("  - Jayson Tatum")
    print("  - Damian Lillard")
    print("  - Anthony Davis")
    print()

    # Get input
    player_name = input("Enter player name (or press Enter for LeBron James): ").strip()
    if not player_name:
        player_name = "LeBron James"

    game_date = input("Enter game date (YYYY-MM-DD, or press Enter for 2025-01-20): ").strip()
    if not game_date:
        game_date = "2025-01-20"

    opponent = input("Enter opponent team ID (or press Enter for 1610612744/Warriors): ").strip()
    if not opponent:
        opponent_team_id = 1610612744
    else:
        try:
            opponent_team_id = int(opponent)
        except:
            print("Invalid team ID, using default")
            opponent_team_id = 1610612744

    home_input = input("Home game? (y/n, or press Enter for away): ").strip().lower()
    is_home = home_input == 'y'

    # Make prediction
    print(f"\n{'='*60}")
    print("MAKING PREDICTION...")
    print(f"{'='*60}\n")

    prediction = predictor.predict_game(
        player_name=player_name,
        game_date=game_date,
        opponent_team_id=opponent_team_id,
        is_home=is_home
    )

    if prediction:
        home_away = "vs" if prediction['is_home'] else "@"
        print(f"🏀 {prediction['player_name']} {home_away} Team {prediction['opponent_team_id']}")
        print(f"📅 {prediction['game_date']}")
        print()
        print("📊 PREDICTED STATS:")
        print(f"   Points:   {prediction['predicted_points']:.1f}")
        print(f"   Rebounds: {prediction['predicted_rebounds']:.1f}")
        print(f"   Assists:  {prediction['predicted_assists']:.1f}")
        print()
        print("✅ Prediction complete!")
    else:
        print("❌ Could not make prediction")
        print("   Possible reasons:")
        print("   - Player not found")
        print("   - Insufficient game history")
        print("   - Invalid inputs")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()

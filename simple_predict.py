#!/usr/bin/env python3
"""
Simple prediction script - Easy to use NBA stat predictor

Usage:
    python3 simple_predict.py
"""

from prediction.predictor import GamePredictor


def main():
    print("=" * 60)
    print("NBA STAT PREDICTOR - SIMPLE VERSION")
    print("=" * 60)
    print()

    # Initialize predictor
    print("Loading models...")
    predictor = GamePredictor()
    print()

    # Popular players to predict
    players_to_predict = [
        {
            'name': 'LeBron James',
            'date': '2025-01-20',
            'opponent': 1610612744,  # Warriors
            'home': False
        },
        {
            'name': 'Stephen Curry',
            'date': '2025-01-20',
            'opponent': 1610612747,  # Lakers
            'home': True
        },
        {
            'name': 'Giannis Antetokounmpo',
            'date': '2025-01-20',
            'opponent': 1610612738,  # Celtics
            'home': True
        },
        {
            'name': 'Kevin Durant',
            'date': '2025-01-20',
            'opponent': 1610612751,  # Nets
            'home': False
        }
    ]

    print("Making predictions...")
    print()
    print("=" * 60)

    for player in players_to_predict:
        pred = predictor.predict_game(
            player_name=player['name'],
            game_date=player['date'],
            opponent_team_id=player['opponent'],
            is_home=player['home']
        )

        if pred:
            location = "vs" if pred['is_home'] else "@"
            print(f"\n{pred['player_name']} {location} Team {pred['opponent_team_id']}")
            print(f"Date: {pred['game_date']}")
            print(f"Prediction: {pred['predicted_points']:.1f} pts | "
                  f"{pred['predicted_rebounds']:.1f} reb | "
                  f"{pred['predicted_assists']:.1f} ast")
            print("-" * 60)
        else:
            print(f"\n{player['name']}: Could not make prediction")
            print("-" * 60)

    print("\n" + "=" * 60)
    print("All predictions complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

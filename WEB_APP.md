# NBA Predictions Web App

Full-stack web application with FastAPI backend and modern frontend.

## Features

- **Upcoming Games**: View all upcoming NBA games
- **Player Predictions**: Get AI predictions for all players in a game
- **Player Search**: Search and filter players
- **Model Performance**: See predicted vs actual stats (test set games)
- **Live Updates**: Real-time NBA schedule integration

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn python-multipart
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Run Locally

```bash
python3 run_api.py
```

The app will be available at:
- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## API Endpoints

### Health
```
GET /api/health
```

### Games
```
GET /api/games/today
GET /api/games/upcoming?days=7
```

### Predictions
```
GET /api/predictions/game/{game_id}
GET /api/predictions/player/{player_id}?game_date=...&opponent_team_id=...&is_home=true
```

### Search
```
GET /api/players/search?query=LeBron
```

### Evaluation
```
GET /api/evaluation/comparison?limit=100
GET /api/stats/summary
```

## Deploy to Vercel

### 1. Install Vercel CLI
```bash
npm install -g vercel
```

### 2. Deploy
```bash
vercel
```

Follow the prompts:
- Project name: `nba-predictions`
- Framework: `Other`
- Build command: (leave empty)
- Output directory: `static`

### 3. Set Environment Variables (if needed)

On Vercel dashboard, add:
- `PYTHON_VERSION`: `3.11`

### 4. Access Your App

Vercel will provide a URL like:
```
https://nba-predictions.vercel.app
```

## Project Structure

```
nba_ml/
├── api/
│   ├── main.py                 # FastAPI app
│   ├── nba_schedule.py         # NBA API integration
│   └── model_evaluation.py     # Predicted vs actual
│
├── static/
│   ├── index.html              # Frontend UI
│   ├── style.css               # Styling
│   └── app.js                  # JavaScript logic
│
├── vercel.json                 # Vercel config
├── run_api.py                  # Local dev server
└── requirements.txt            # Python dependencies
```

## Features Explained

### Upcoming Games
- Fetches live NBA schedule from NBA API
- Shows next 7 days of games
- Click any game to see player predictions

### Player Predictions
- Enter a game ID
- See predictions for ALL players in that game
- Separated by home/away teams
- Shows points, rebounds, assists predictions

### Player Search
- Search by name (min 2 characters)
- Real-time search as you type
- Returns player ID and name

### Model Performance
- Shows predicted vs actual stats
- Only uses games NOT in training data (test set)
- Color-coded errors:
  - Green: Low error (< 3)
  - Orange: Medium error (3-6)
  - Red: High error (> 6)

## Customization

### Change Number of Days
Edit `api/main.py`:
```python
@app.get("/api/games/upcoming")
async def get_upcoming(days: int = 14):  # Change from 7 to 14
```

### Change Comparison Limit
Edit `static/app.js`:
```javascript
const response = await fetch(`${API_BASE}/api/evaluation/comparison?limit=200`);
```

### Change Model
The API automatically uses the latest XGBoost model. To force LightGBM:
```python
# In api/main.py
predictor = GamePredictor(model_version='lightgbm_v1.0.0_...')
```

## Troubleshooting

**Port already in use:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

**Database not found:**
```bash
# Make sure you ran setup
python3 setup_from_scratch.py
```

**No predictions showing:**
- Check that models are trained
- Verify database has recent data
- Check browser console for errors

## Performance

- **API Response Time**: < 200ms per prediction
- **Batch Predictions**: ~30 players in < 2 seconds
- **Database Queries**: Optimized with indexes

## Next Steps

- [ ] Add real-time game updates
- [ ] Add player comparison tool
- [ ] Add confidence intervals
- [ ] Add historical performance charts
- [ ] Add team predictions

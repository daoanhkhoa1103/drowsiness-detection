# Driver & Drowsiness Management System

## Features
- Manage drivers (add, edit, view)
- Manage vehicles (add, edit, view)
- Manage routes (add, edit, view)
- Real-time drowsiness detection with webcam, warning banner and sound
- Beautiful Bootstrap interface
- Sample data for quick demo

## How to use
1. Install dependencies
2. Place your trained model file (`best_model_2.pth`) and `voice.mp3` into the project root (or as directed in `app.py`)
3. Run app:
4. Visit [http://localhost:8000](http://localhost:8000) in your browser.

## Folders
- `templates/` - HTML files
- `static/` - voice.mp3 for warning sound
- `database.db` - SQLite database (provided sample)
- `model.py` - CNN class

## Notes
- Requires a webcam for drowsiness detection
- All admin/data management features available from web dashboard

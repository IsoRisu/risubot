
# Setting Up the Bot

Follow these steps to set up and run the bot:

## 1. Create a `botkey.py` File

Create a new file named `botkey.py` and store your bot's key as a string in the variable `bot_key`:

```python
# botkey.py
bot_key = "your-bot-key-here"
```

## 2. Create a Virtual Environment

Create a virtual environment to isolate your project dependencies:

```bash
python -m venv venv
```

## 3. Install Dependencies

Activate the virtual environment and install the required dependencies from `requirements.txt`:

- On **Windows**:

  ```bash
  .\venv\Scripts\activate
  ```

- On **Linux**:

  ```bash
  source venv/bin/activate
  ```

Then, install the required packages:

```bash
pip install -r requirements.txt
```

## 4. Run the App

Once the dependencies are installed, run the bot application:

```bash
python risubot.py
```

Your bot should now be up and running!

---


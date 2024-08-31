import asyncpraw
import asyncio
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd

# Define your credentials
client_id = '5BPI-DSXrAjAkH7wMfSIGw'
client_secret = 'UMoH9BQh3zCpgj08Ez0AYxYo6-04Lg'
user_agent = 'CustomerFeedbackAnalysis by Wise_Whole3908'

# Database setup
DATABASE_URL = "sqlite:///reddit_data.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define data model
class RedditPost(Base):
    __tablename__ = "reddit_posts"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    score = Column(Integer)
    url = Column(String)
    created_utc = Column(Float)

Base.metadata.create_all(bind=engine)

# Create an async Reddit client
async def fetch_reddit_data():
    async with asyncpraw.Reddit(client_id=client_id,
                                client_secret=client_secret,
                                user_agent=user_agent) as reddit:
        subreddit = await reddit.subreddit('python')
        posts = []
        async for submission in subreddit.top(limit=10):
            posts.append({
                'title': submission.title,
                'score': submission.score,
                'url': submission.url,
                'created_utc': submission.created_utc
            })
        return posts

# Function to save data to SQLite
def save_to_database(data):
    session = SessionLocal()
    try:
        for post in data:
            db_post = RedditPost(
                title=post['title'],
                score=post['score'],
                url=post['url'],
                created_utc=post['created_utc']
            )
            session.add(db_post)
        session.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        session.rollback()
    finally:
        session.close()

# Run the async function and save data
async def main():
    data = await fetch_reddit_data()
    loop.run_in_executor(None, save_to_database, data)

# Create a new event loop and run the main function
if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())

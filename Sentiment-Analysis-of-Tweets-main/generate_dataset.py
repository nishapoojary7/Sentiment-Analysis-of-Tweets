import os
import sys
import random
import csv

# Set UTF-8 encoding for stdout to prevent Windows console encoding errors
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Set up paths
workspace_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(workspace_dir, "dataset")
os.makedirs(dataset_dir, exist_ok=True)
dataset_path = os.path.join(dataset_dir, "tweets.csv")

# Word lists for generating tweets
products = ["phone", "laptop", "app", "keyboard", "headphones", "camera", "watch", "software", "service", "customer support", "flight", "restaurant", "food", "hotel", "game", "movie", "book"]
companies = ["Amazon", "Google", "Apple", "Microsoft", "Netflix", "Uber", "Airbnb", "Samsung", "Dell", "Sony", "Steam"]
cities = ["New York", "London", "Tokyo", "Paris", "Seattle", "San Francisco", "Sydney", "Mumbai", "Berlin"]

pos_adj = ["amazing", "awesome", "fantastic", "incredible", "beautiful", "wonderful", "outstanding", "excellent", "superb", "brilliant", "perfect", "delicious", "friendly", "fast", "reliable", "good", "great", "nice", "happy", "glad", "helpful"]
neg_adj = ["terrible", "worst", "horrible", "awful", "useless", "broken", "waste of money", "disappointing", "annoying", "slow", "laggy", "buggy", "frustrating", "rude", "poor", "bad", "stupid", "hate", "dumb", "rubbish", "crap", "sucks", "annoyed"]

pos_emojis = ["😊", "😍", "🎉", "🔥", "🙌", "💖", "👍", "🚀", "✨"]
neg_emojis = ["😞", "😠", "😡", "😢", "👎", "🤮", "💩", "💔", "🤦"]

pos_hashtags = ["#happy", "#loveit", "#excellent", "#greatservice", "#winning", "#success", "#recommend", "#perfect", "#amazing", "#satisfied"]
neg_hashtags = ["#fail", "#wasteofmoney", "#frustrated", "#disappointed", "#worstservice", "#neveragain", "#terrible", "#buggy", "#annoyed", "#poorquality"]
neut_hashtags = ["#updates", "#tech", "#news", "#review", "#opinion", "#daily", "#random", "#reading", "#honestopinion", "#life"]

# Tweet templates
pos_templates = [
    "I love my new {product}! The performance is absolutely {pos_adj}. {emoji} {hashtag}",
    "This is the best {product} I have ever bought! Highly recommend it to everyone. {emoji}",
    "Amazing customer service from @{company}. They resolved my issue in minutes! {hashtag}",
    "Super happy with the performance of this {product}. Worth every penny! {emoji} {hashtag}",
    "Delicious food, friendly staff, and great ambiance at the new restaurant. Had a fantastic night! {emoji}",
    "The concert tonight was incredible! The energy was out of this world. {emoji} {hashtag}",
    "So excited to start my new job tomorrow! Feeling blessed and ready. {emoji} {hashtag}",
    "The flight was on time and the crew was very {pos_adj}. Great experience with @{company}! {emoji}",
    "Just received my order and it is in {pos_adj} condition. Fast delivery too! {hashtag}",
    "Absolutely {pos_adj} update! The interface is so smooth now. Well done! {emoji} {hashtag}",
    "Highly impressed by the new features of this {product}. It makes my life so much easier. {emoji}",
    "The customer care team at @{company} was so {pos_adj} today. Thank you for the quick help! {emoji} {hashtag}",
    "Had a {pos_adj} weekend getaway in {city}. The hotel was perfect! {emoji}",
    "This {product} has {pos_adj} battery life. Can easily go two days without charging. {hashtag}",
    "This is a very good {product}. I enjoy using it and feel happy. {emoji}",
    "I have a very good feeling about this. Highly recommended! {emoji} {hashtag}",
    "The cinematography in the new movie is absolutely {pos_adj}. A must-watch! {emoji}"
]

neg_templates = [
    "This {product} is the worst! The battery life is totally {neg_adj}. {emoji} {hashtag}",
    "Terrible customer support from @{company}. They were extremely rude and unhelpful. {emoji} {hashtag}",
    "Very disappointed with my purchase. The {product} arrived broken and is a complete waste of money. {emoji}",
    "I hate the new software update for my {product}! It keeps crashing and freezing. {emoji} {hashtag}",
    "The service here is extremely slow. Avoid this place if you can! {emoji}",
    "Worst experience ever. The food was cold, tasteless, and overpriced. {emoji} {hashtag}",
    "My package got delayed again! @{company} has the worst delivery service. {emoji} {hashtag}",
    "The camera quality on this new {product} is extremely {neg_adj}. Total disappointment. {hashtag}",
    "This is a stupid app. It is completely useless and slow. {emoji} {hashtag}",
    "I hate this {product}! It is so bad and frustrating. {emoji} {hashtag}",
    "The keyboard has a terrible typing experience. The keys feel mushy and {neg_adj}. {emoji}",
    "I tried contacting customer support but they just ignored my request. Stupid service! {emoji}",
    "Had a {neg_adj} stay at the hotel in {city}. The room was dirty and noisy. {emoji} {hashtag}",
    "This {product} is so overrated. It has a {neg_adj} build quality and feels cheap. {emoji} {hashtag}"
]

neut_templates = [
    "Just finished reading my new book. It was an okay read, nothing special. {hashtag}",
    "The package for my new {product} arrived today from @{company}.",
    "I am going to the grocery store now to buy some food and household items.",
    "It is raining today in {city}. Just a normal rainy day. {hashtag}",
    "Here is my honest review of the new {product}. It is average, has both pros and cons.",
    "Does anyone know if the meeting is still scheduled for tomorrow morning?",
    "Using the new {product} for my daily work. It does the job, nothing more, nothing less.",
    "Watching a documentary about {city} on @{company} Netflix tonight.",
    "I have been using this {product} for six months. It is decent for the price.",
    "Just saw the news about @{company}'s new product release. Let's see how it performs.",
    "The bus was a bit delayed today, but that's normal for {city} traffic. {hashtag}",
    "Comparing the specifications of the new {product} with the older model.",
    "Received an update notification for the @{company} app today.",
    "The restaurant in {city} was not bad, but it wasn't great either. Just average. {hashtag}",
    "Looking for recommendations for a standard {product} under a reasonable budget."
]

def generate_tweets(count=1500):
    dataset = []
    
    # Target counts to keep it balanced
    pos_count = count // 3
    neg_count = count // 3
    neut_count = count - pos_count - neg_count
    
    # Generate Positive Tweets
    for _ in range(pos_count):
        template = random.choice(pos_templates)
        tweet = template.format(
            product=random.choice(products),
            pos_adj=random.choice(pos_adj),
            emoji=random.choice(pos_emojis),
            company=random.choice(companies),
            city=random.choice(cities),
            hashtag=random.choice(pos_hashtags)
        )
        dataset.append((tweet, "Positive"))
        
    # Generate Negative Tweets
    for _ in range(neg_count):
        template = random.choice(neg_templates)
        tweet = template.format(
            product=random.choice(products),
            neg_adj=random.choice(neg_adj),
            emoji=random.choice(neg_emojis),
            company=random.choice(companies),
            city=random.choice(cities),
            hashtag=random.choice(neg_hashtags)
        )
        dataset.append((tweet, "Negative"))
        
    # Generate Neutral Tweets
    for _ in range(neut_count):
        template = random.choice(neut_templates)
        tweet = template.format(
            product=random.choice(products),
            company=random.choice(companies),
            city=random.choice(cities),
            hashtag=random.choice(neut_hashtags)
        )
        dataset.append((tweet, "Neutral"))
        
    # Shuffle the dataset
    random.shuffle(dataset)
    return dataset

def main():
    print(f"Generating synthetic dataset of tweets...")
    tweets = generate_tweets(1500)
    
    with open(dataset_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Tweet", "Sentiment"])
        writer.writerows(tweets)
        
    print(f"Dataset successfully saved to: {dataset_path}")
    print(f"Total rows generated: {len(tweets)}")
    
    # Inspect first few rows
    print("\nFirst 5 rows of generated dataset:")
    for i, (tweet, sentiment) in enumerate(tweets[:5]):
        print(f"{i+1}. [{sentiment}] {tweet}")

if __name__ == "__main__":
    main()

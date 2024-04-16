import pandas as pd
import folium
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from folium.plugins import MarkerCluster

# Define the path to the Excel file
file_path = r"D:\Dataset .xlsx"

# Load the dataset using the file path
df = pd.read_excel('D:\Dataset .xlsx')

# Assuming the column containing cuisines is named 'Cuisines'
# Count the occurrences of each cuisine
cuisine_counts = df['Cuisines'].value_counts()

# Get the top three most common cuisines
top_cuisines = cuisine_counts.head(3)

# Calculate the percentage of restaurants that serve each of the top cuisines
total_restaurants = df.shape[0]
top_cuisines_percentage = (top_cuisines / total_restaurants) * 100

# Print the results
print("Top three most common cuisines and the percentage of restaurants that serve them:")
for cuisine, percentage in top_cuisines_percentage.items():
    print(f"{cuisine}: {percentage:.2f}%")
    #task 2
    # Count the occurrences of each city
    city_counts = df['City'].value_counts()

    # Find the city with the highest number of restaurants
    city_with_most_restaurants = city_counts.idxmax()
    most_restaurants_count = city_counts.max()

    # Print the city with the highest number of restaurants
    print(
        f"The city with the highest number of restaurants is '{city_with_most_restaurants}' with {most_restaurants_count} restaurants.")
    #task 3
# Count the occurrences of each price range category
price_range_counts = df['Price range'].value_counts()

# Create a bar chart to visualize the distribution of price ranges
plt.figure(figsize=(8, 6))
price_range_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Price ranges Among Restaurants')
plt.xlabel('Price range')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Calculate the percentage of restaurants in each price range category
total_restaurants = len(df)
price_range_percentages = (price_range_counts / total_restaurants) * 100

# Print the percentage of restaurants in each price range category
print("Percentage of restaurants in each price range category:")
print(price_range_percentages)
#task 4
# Convert string values to boolean
df['Has Online delivery'] = df['Has Online delivery'].map({'Yes': True, 'No': False})

# Determine the percentage of restaurants that offer online delivery
total_restaurants = len(df)
restaurants_with_delivery = df['Has Online delivery'].sum()
percentage_with_delivery = (restaurants_with_delivery / total_restaurants) * 100

print(f"Percentage of restaurants that offer online delivery: {percentage_with_delivery:.2f}%")
#Level 2 Task 1
# Analyze the distribution of aggregate ratings
rating_counts = df['Aggregate rating'].value_counts().sort_index()

# Determine the most common rating range
most_common_rating = rating_counts.idxmax()

# Calculate the average number of votes received by restaurants
average_votes = df['Votes'].mean()

# Print the results
print("Distribution of Aggregate Ratings:")
print(rating_counts)
print(f"Most common rating range: {most_common_rating}")
print(f"Average number of votes received by restaurants: {average_votes:.2f}")
#task 2
# Split the 'Cuisines' column into separate rows
df = df.assign(Cuisine=df['Cuisines'].str.split(', ')).explode('Cuisine')

# Determine the most common combinations of cuisines
cuisine_counts = df['Cuisine'].value_counts()

# Print the most common combinations of cuisines
print("Most common combinations of cuisines:")
print(cuisine_counts.head(10))  # Adjust the number as needed

# Determine if certain cuisine combinations tend to have higher ratings
# Analyze the average rating for each cuisine combination
average_ratings = df.groupby('Cuisine')['Aggregate rating'].mean().sort_values(ascending=False)

# Print the top-rated cuisine combinations
print("\nTop-rated cuisine combinations:")
print(average_ratings.head(10))  # Adjust the number as needed
#task 3
# Create a map centered at the mean latitude and longitude of all restaurants
center_lat = df['Latitude'].mean()
center_lon = df['Longitude'].mean()
restaurant_map = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# Create a MarkerCluster layer
marker_cluster = MarkerCluster().add_to(restaurant_map)

# Add markers for each restaurant location
for index, row in df.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], popup=row['Restaurant Name']).add_to(marker_cluster)

# Display the map
restaurant_map.save("restaurant_map.html")  # Save the map as an HTML file
restaurant_map

# Alternatively, you can save the map as a PNG image using matplotlib
# plt.figure(figsize=(10, 8))
# plt.imshow(restaurant_map)
# plt.axis('off')
# plt.savefig("restaurant_map.png", bbox_inches='tight', pad_inches=0)
# plt.show()
#task 4
# Identify restaurant chains
chain_counts = df['Restaurant Name'].value_counts()
chains = chain_counts[chain_counts > 1].index.tolist()

if chains:
    print("Restaurant chains present in the dataset:")
    print(chains)
else:
    print("No restaurant chains found in the dataset.")

# Analyze the ratings and popularity of different restaurant chains
chain_ratings = {}
chain_votes = {}
for chain in chains:
    chain_df = df[df['Restaurant Name'] == chain]
    chain_ratings[chain] = chain_df['Aggregate rating'].mean()
    chain_votes[chain] = chain_df['Votes'].sum()

# Sort chains by average rating and print the results
sorted_chains_by_rating = sorted(chain_ratings.items(), key=lambda x: x[1], reverse=True)
print("\nChains sorted by average rating:")
for chain, rating in sorted_chains_by_rating:
    print(f"{chain}: Average Rating - {rating:.2f}")

# Sort chains by total votes and print the results
sorted_chains_by_votes = sorted(chain_votes.items(), key=lambda x: x[1], reverse=True)
print("\nChains sorted by total votes:")
for chain, votes in sorted_chains_by_votes:
    print(f"{chain}: Total Votes - {votes}")
#Level 3 Task 1

# Tokenize the rating text
rating_text = df['Rating text'].dropna()
all_words = ' '.join(rating_text)
tokens = word_tokenize(all_words)

# Perform sentiment analysis
sia = SentimentIntensityAnalyzer()
positive_words = [word for word in tokens if sia.polarity_scores(word)['compound'] > 0]
negative_words = [word for word in tokens if sia.polarity_scores(word)['compound'] < 0]

# Count occurrences of positive and negative keywords
positive_counts = pd.Series(positive_words).value_counts().head(10)
negative_counts = pd.Series(negative_words).value_counts().head(10)

# Print the most common positive and negative keywords
print("Most common positive keywords:")
print(positive_counts)
print("\nMost common negative keywords:")
print(negative_counts)

# Calculate the average length of rating text
df['Rating Text Length'] = df['Rating text'].str.len()
average_rating_text_length = df['Rating Text Length'].mean()
print(f"\nAverage rating text length: {average_rating_text_length:.2f} characters")

# Explore the relationship between rating text length and rating
plt.scatter(df['Rating Text Length'], df['Aggregate rating'])
plt.title('Relationship between Rating Text Length and Rating')
plt.xlabel('Rating Text Length')
plt.ylabel('Aggregate Rating')
plt.show()
#task2
# Identify restaurants with the highest and lowest number of votes
restaurant_highest_votes = df.loc[df['Votes'].idxmax()]
restaurant_lowest_votes = df.loc[df['Votes'].idxmin()]

print("Restaurant with the highest number of votes:")
print(restaurant_highest_votes[['Restaurant Name', 'Votes']])

print("\nRestaurant with the lowest number of votes:")
print(restaurant_lowest_votes[['Restaurant Name', 'Votes']])

# Analyze the correlation between number of votes and rating
correlation = df['Votes'].corr(df['Aggregate rating'])
print(f"\nCorrelation between number of votes and rating: {correlation:.2f}")

# Scatter plot for number of votes vs. rating
plt.scatter(df['Votes'], df['Aggregate rating'])
plt.title('Number of Votes vs. Rating')
plt.xlabel('Number of Votes')
plt.ylabel('Aggregate Rating')
plt.show()
#task 3
# Ensure the 'Has Online delivery' and 'Has Table booking' columns are of a numeric type
df['Has Online delivery'] = pd.to_numeric(df['Has Online delivery'], errors='coerce')
df['Has Table booking'] = pd.to_numeric(df['Has Table booking'], errors='coerce')

# Group the data by 'Price range'
price_range_groups = df.groupby('Price range')

# Calculate the proportion of restaurants offering online delivery and table booking in each price range
proportion_delivery = price_range_groups['Has Online delivery'].mean()
proportion_table_booking = price_range_groups['Has Table booking'].mean()

# Plotting
plt.figure(figsize=(10, 5))

# Proportion of restaurants offering online delivery by price range
plt.subplot(1, 2, 1)
proportion_delivery.plot(kind='bar', color='skyblue')
plt.title('Proportion of Restaurants Offering Online Delivery by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Proportion with Online Delivery')
plt.xticks(rotation=0)

# Proportion of restaurants offering table booking by price range
plt.subplot(1, 2, 2)
proportion_table_booking.plot(kind='bar', color='salmon')
plt.title('Proportion of Restaurants Offering Table Booking by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Proportion with Table Booking')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()




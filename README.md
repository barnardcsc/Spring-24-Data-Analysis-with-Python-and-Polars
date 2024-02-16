# Data Analysis with Python & Polars

This workshop focuses on essential data analysis skills using Python and Polars. We'll cover fetching and wrangling data, basic data cleaning, descriptive statistics, visualization, and basic text analysis. The aim is to enhance your Python data analysis abilities, with universally applicable techniques for various datasets and fields.

**Polars?**

This workshop introduces Polars rather than the more common Python data science stack (Pandas & Numpy):

- **Faster**: Built to be fast and can perform common operations 5–10 times faster than Pandas. Since its more memory-efficient, it requires less RAM compared to Pandas as well.
- **Growing in Usage**: Gaining traction among data professionals for its versatility & performance.
- **Friendly(er) API**: Data manipulation is simpler and more intuitive than Pandas; code is more readable.

## Data Source

- **About the Data**: [Dataset Information](https://data.cityofnewyork.us/City-Government/Jobs-NYC-Postings/kpav-sd4t/about_data)
- **JSON Source**: [Jobs NYC Postings JSON](https://data.cityofnewyork.us/resource/kpav-sd4t.json)

## Learning Outcomes

In this workshop, you will:

- **Fetch and Clean Data**: Retrieve data from NYC's job postings API and prepare it for analysis.
- **Explore Data**: Perform exploratory data analysis including data grouping, filtering, and aggregation.
- **Visualize Insights**: Create charts using Matplotlib to visualize job data trends.
- **Process Text Data**: Generate word clouds from job descriptions for text analysis.
- **Export Findings**: Save processed data to CSV for external use.

This workshop aims to equip you with key data analysis skills, focusing on practical applications using real-world data.

Please feel free to ask questions or work in teams.

## Contact
You can reach the CSC at csc@barnard.edu or reach out to Marko Krkeljas at mkrkelja@barnard.edu.

## Setting Up

For this workshop, we'll be using [Google Colab Notebooks](https://colab.research.google.com/). 

**What is a Google Colab Notebook?**

  - Google Colab lets you write and execute Python code in your browser.
  - No setup required; just a Google account.

**Creating a Google Colab Notebook**

  1. [Follow this link.](https://colab.research.google.com/)
  2. Click on the "+ New notebook" button on the bottom-left side.

> &#x26a0;&#xfe0f; **For this workshop, we highly recommend using Google Colab, as we can't troubleshoot installation issues.**

## Workshop

### Dataset

- [Jobs NYC Postings](https://data.cityofnewyork.us/City-Government/Jobs-NYC-Postings/kpav-sd4t/about_data)
- [API Endpoint](https://data.cityofnewyork.us/resource/kpav-sd4t.json)

Potential Focus Areas:

1. **Salary Trends**: Analyze pay scales across positions, offering insights into equity and economic impact. Inflation? 

2. **Employment Dynamics**: Track job posting trends over time to identify shifts in government focus.

3. **Geographical Insights**: Assess job distribution across boroughs, revealing spatial employment patterns.

4. **Qualification Requirements**: Examine necessary skills and qualifications, indicating educational and experience standards.

5. **Policy Impact**: Investigate how policy changes influence job postings.


### Intro to Python (Data Types, Functions, Loops)

This workshop assumes basic knowledge of Python. For a brief review, please refer to the Python tutorial in the introductory section provided below.

- [Fall-23-Data-Analysis-With-Python-Halloween-Edition
](https://github.com/barnardcsc/Fall-23-Data-Analysis-With-Python-Halloween-Edition)


### Fetching Data

Computer programs communicate with servers via `HTTP` requests to get or send data. This data is often shared in formats like JSON. 

**JSON**: JavaScript Object Notation

- Lightweight data-interchange format.
- Human-readable and easy for machines to parse.

```json
[
	{
	  "name": "Jerry",
	  "dob": "1977-05-08",
	},
	{
	  "name": "Donna",
	  "dob": "1971-12-31",
	},
]
```

---

**Fetching the data (Attempt #1)**: 

To fetch the data, we make a request to the endpoint/URL:

```python
import requests
import polars as pl

url = "https://data.cityofnewyork.us/resource/kpav-sd4t.json"
response = requests.get(url)
data = response.json()
df = pl.DataFrame(data)
len(df)
```

> &#x26a0;&#xfe0f; This might throw an error due to the I/O limits on Colab Notebooks. If it doesn't throw, notice that the data set only has 1000 rows.


**Fetching the data (Attempt #2)**: 

To get all of the data, we have to paginate through it:

```python
import requests
import polars as pl

def fetch_all_nyc_jobs_data():
  base_url = "https://data.cityofnewyork.us/resource/kpav-sd4t.json"
  data = []
  offset = 0
  limit = 1000  # Fetch 1000 rows at a time

  while True:
      # Construct the URL with offset for pagination
      url = f"{base_url}?$limit={limit}&$offset={offset}"
      response = requests.get(url)
      batch = response.json()

      # Break the loop if no more data is returned
      if not batch:
          break

      data.extend(batch)
      offset += limit

  # Convert the fetched data to a Polars DataFrame
  df = pl.DataFrame(data)
  return df

# Fetch data and display the first few rows
df = fetch_all_nyc_jobs_data()
df.head()
```

### Cleaning the Data

**Checking Data Types of Each Column**

Before you begin analysing the data, start by checking the column data types.

```python
print(df.dtypes)  # Note: Initially, all columns are strings.
```

**Why Using `describe` Might Not Be Useful Here**

`describe` provides summary statistics, but it's not useful when the column types are wrong.

```python
print(df.describe())
```

**Data Type Conversion**

Convert columns to appropriate data types.

```python
df = df.with_columns([
    pl.col('number_of_positions').cast(pl.Int32),  # Convert to integer
    pl.col('salary_range_from').cast(pl.Float64).round().cast(pl.Int32),  # Convert to integer after rounding
    pl.col('salary_range_to').cast(pl.Float64).round().cast(pl.Int32),  # Convert to integer after rounding
    pl.col('posting_date').str.strptime(pl.Date, "%Y-%m-%dT%H:%M:%S%.f"),  # Convert to date with correct fractional seconds
    pl.col('posting_updated').str.strptime(pl.Date, "%Y-%m-%dT%H:%M:%S%.f"),  # Convert to date with correct fractional seconds
    pl.col('process_date').str.strptime(pl.Date, "%Y-%m-%dT%H:%M:%S%.f"),  # Convert to date with correct fractional seconds
    pl.col('post_until').str.strptime(pl.Date, "%d-%b-%Y")  # Convert 'post_until' to date with correct format
])
```

**Display Updated Data Types**

```python
print(df.dtypes)
df.head()
```

**Additional Data Cleaning**

- Identify and address null/missing values.
- Ensure consistent formatting across columns.
- Remove or impute irrelevant or duplicate data.


### Data Wrangling

**Unique Values in Categorical Columns**

```python
df['agency'].unique()
```

**Filtering**

Filtering rows where `number_of_positions` is greater than 5.

```python
filtered_df = df.filter(pl.col('number_of_positions') > 5)
print(filtered_df)
```

**Grouping and Aggregation**

Aggregation helps summarize data for better understanding. In this case, grouping by `agency` and counting job postings gives a clear picture of each agency's job distribution.

```python
grouped_df = df.group_by('agency').agg([
    pl.count('agency').alias('Number of Jobs')
])
print(grouped_df)
```

**Sorting Data**

Sorting by `Total Positions` in descending order to find agencies with the most job openings.

```python
sorted_df = df.group_by('agency').agg([
    pl.sum('number_of_positions').alias('Total Positions')
]).sort('Total Positions', descending=True)
print(grouped_df.head())
```

**Unique Career Levels**

Research areas:

- How do job qualifications change at different career levels?
- What does this say about how skills are valued?
- What role does seniority/experience play into minimal requirements, preferred skills, salary?

```python
unique_career_levels = df['career_level'].unique()
print(unique_career_levels)
```

**Filtering Data Based on Multiple Conditions**

Filtering for student jobs that are `internal` postings.

```python
df_students = df.filter(
    (pl.col('career_level') == 'Student') &
    (pl.col('posting_type') == 'Internal')
)
print(len(df_students))
print(df_students.head())
```

**Grouping and Sorting**

Grouping student jobs by `job_category` and sorting by the number of jobs.

```python
df_students_grouped = df_students.group_by('job_category').agg([
    pl.count().alias('Number of Jobs')
]).sort('Number of Jobs', descending=True)
print(df_students_grouped)
```

### Visualization

Visualizing the number of student jobs by job category.

```python
import matplotlib.pyplot as plt

df_students_pd = df_students_grouped.to_pandas()
plt.figure(figsize=(10, 8))
plt.barh(df_students_pd['job_category'], df_students_pd['Number of Jobs'], color='skyblue')
plt.xlabel('Number of Jobs')
plt.ylabel('Job Category')
plt.title('Number of Jobs by Job Category')
plt.gca().invert_yaxis()  # Invert y-axis to have the largest bar on top
plt.show()
```

**Time Series Analysis: Jobs Posted Per Month**

First, extract year and month from `posting_date`, then count postings by year and month.

Prepare the DataFrame with year and month columns:

```python
all_jobs = df.with_columns([
    pl.col("posting_date").dt.year().alias("year"),
    pl.col("posting_date").dt.month().alias("month")
])
```

**Group by Year and Month, Then Count Postings**

```python
monthly_counts = all_jobs.group_by(["year", "month"]).agg([
    pl.count().alias("postings_count")
]).sort(["year", "month"])
```

**Convert to Pandas DataFrame for Plotting**

```python
monthly_counts_pd = monthly_counts.to_pandas()

# Add 'Year-Month' column for plotting
monthly_counts_pd["ym"] = (
    monthly_counts_pd["year"].astype(str)
    + "-"
    + monthly_counts_pd["month"].astype(str).str.zfill(2)
)

# zfill() pads a string on the left with zeros to a specified width. 
# Using zfill(2) ensures months are two digits, turning '2' (February) into '02'.

```

**Create the Bar Chart**

```python
plt.figure(figsize=(15, 8))
plt.bar(monthly_counts_pd["ym"], monthly_counts_pd["postings_count"], color="skyblue")
plt.xlabel("Year-Month")
plt.ylabel("Number of Job Postings")
plt.title("Number of Job Postings per Month")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
```


**Discussion & Research Areas**

- Relationship between job postings & passage of 2023 budget?
  - [NYC Budget Financial Plan (June 2023)](https://www.nyc.gov/site/omb/publications/finplan06-23.page)
- Could you correlate budget increase and/or agency budget allocation with job postings? 
- Budget allocation per each new job?
- Can this data be augmented with additional datasets? Do these trends hold for other cities? 

### Salary Analysis

Potential research areas: 

- Pay scales across positions, agencies, experience levels
- Salary ranges/spreads
- Change over time and/or tracking inflation/trends
- Where are salaries growing/stalling?

**Adding columns for average salary and salary range**

```python
df_with_salary_info = df.with_columns([
    ((pl.col('salary_range_from') + pl.col('salary_range_to')) / 2).alias('average_salary'),
    (pl.col('salary_range_to') - pl.col('salary_range_from')).alias('salary_range')
])
df_with_salary_info.head()
```

**Jobs with Largest Salary Spread**

Sorting by `salary_range` to find jobs with the largest spread in salaries.

```python
df_with_salary_info.sort('salary_range', descending=True).select(['civil_service_title', 'job_category', 'salary_range'])
```

**Analyzing Average Salary by Career Level and Title**

Grouping by `career_level` and `title_code_no` to calculate the average salary.

```python
avg_salary_per_career_level = df_with_salary_info.group_by(['career_level', 'title_code_no']).agg([
    pl.mean('average_salary').alias('average_salary_per_level')
]).sort(['title_code_no', 'career_level', 'average_salary_per_level'])
print(avg_salary_per_career_level)
```

**Jobs with Largest Salary Spread (Filtered View)**

Sorting by `salary_range` and selecting specific columns for a clear view.

```python
df_with_salary_info.sort('salary_range', descending=True).select(['civil_service_title', 'job_category', 'salary_range'])
```

**Exporting Data to CSV**

Writing the DataFrame to a CSV for spreadsheet software.

```python
file_path = 'df_with_salary_info.csv'  # Specify your desired file path
df_with_salary_info.write_csv(file_path)
```

### Text Analysis & Word Cloud

- Text columns are difficult to analysis with traditional quantitative methods.
- This example is an introduction to basic natural langauge processing (NLP) methods.
- Visualize key themes with word clouds.
- Tremendous opportunity with using LLMs/AI.

Instal required libraries for text analysis and word cloud generation.

```python
pip install wordcloud nltk
```

**Aggregating `preferred_skills` and Removing Any None Values**

```python
all_skills = ""
for skill in df['preferred_skills'].to_list():
    if skill:  # Check if the skill is not None
        all_skills += skill + " "  # Append the skill to the all_skills string
```

**Setting Up Stopwords and Word Cloud Generation**

```python
import nltk
from nltk.corpus import stopwords
```

**Download NLTK Stopwords**

```python
nltk.download('stopwords')

# Set of English stop words
stop_words = set(stopwords.words('english'))
```

**Function to Filter Out Stop Words from Text**

```python
def remove_stop_words(text):
    filtered_words = []  # Initialize an empty list to store words that are not stop words

    # Split the text into individual words
    words = text.split()

    # Iterate over each word in the text
    for word in words:
        # Check if the word is not a stop word
        if word.lower() not in stop_words:
            # Add the word to the list of filtered words
            filtered_words.append(word)

    # Join the filtered words back into a single string
    filtered_text = " ".join(filtered_words)

    return filtered_text
```

**Filter Out Stop Words from All Preferred Skills**

```python
filtered_skills = remove_stop_words(all_skills)
```

**Create and Display the Word Cloud Using Matplotlib**

```python
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(width = 800, height = 800,
                      background_color ='white',
                      stopwords = stop_words,
                      min_font_size = 10).generate(filtered_skills)

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()
```

### Conclusion

As we conclude, it's worth mentioning that since a significant portion of the data resides within text fields, there's vast potential in augmenting this data with artificial intelligence (AI) and large language models (LLMs). If you're interested in this work, we have [a workshop on building software with AI & LLMs here](https://github.com/barnardcsc/Fall-23-Building-Software-with-AI-Introduction-to-LLM-APIs).

Please reach out to the CSC at csc@barnard.edu or contact Marko Krkeljas at mkrkelja@barnard.edu if you have any questions.

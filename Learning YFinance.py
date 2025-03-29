# -------------------------------------------------------------------------------------
# Step 1: Install Libraries (for completeness, but you have them commented out)
# -------------------------------------------------------------------------------------
# !pip install yfinance
# !pip install pandas
# !pip install plotly
# !pip install tabulate
# -------------------------------------------------------------------------------------

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from tabulate import tabulate

# -------------------------------------------------------------------------------------
# Step 2: Functions for Data Retrieval and Plotting
# -------------------------------------------------------------------------------------

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return stock

def plot_candlestick_chart(stock):
    # Get historical data for the stock
    hist = stock.history(period="1y")  # Show 1 year of data by default
    fig = go.Figure(data=[go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        increasing_line_color='green', decreasing_line_color='red'
    )])

    fig.update_layout(
        title='Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False
    )

    fig.show()

def display_stock_info(stock):
    info = stock.info
    df1 = pd.DataFrame(info.items(), columns=["Key", "Value"])
    print("### Stock Information")
    print(tabulate(df1, headers='keys', tablefmt='pretty', showindex=False))

def display_stock_actions(stock):
    print("### Stock Actions (Dividends & Splits)")
    print(tabulate(stock.actions, headers='keys', tablefmt='pretty', showindex=False))

def display_stock_financials(stock):
    print("### Stock Financials")
    print(tabulate(stock.financials, headers='keys', tablefmt='pretty', showindex=True))

def display_stock_quarterly_financials(stock):
    print("### Stock Quarterly Financials")
    print(tabulate(stock.quarterly_financials, headers='keys', tablefmt='pretty', showindex=True))

def display_major_holders(stock):
    print("### Major Holders")
    print(tabulate(stock.major_holders, headers='keys', tablefmt='pretty', showindex=True))

def display_institutional_holders(stock):
    print("### Institutional Holders")
    print(tabulate(stock.institutional_holders, headers='keys', tablefmt='pretty', showindex=True))

def display_balance_sheet(stock):
    print("### Balance Sheet")
    print(tabulate(stock.balance_sheet, headers='keys', tablefmt='pretty', showindex=True))

def display_quarterly_balance_sheet(stock):
    print("### Quarterly Balance Sheet")
    print(tabulate(stock.quarterly_balance_sheet, headers='keys', tablefmt='pretty', showindex=True))

def display_cashflow(stock):
    print("### Cashflow")
    print(tabulate(stock.cashflow, headers='keys', tablefmt='pretty', showindex=True))

def display_quarterly_cashflow(stock):
    print("### Quarterly Cashflow")
    print(tabulate(stock.quarterly_cashflow, headers='keys', tablefmt='pretty', showindex=True))

def display_earnings(stock):
    print("### Earnings")
    print(tabulate(stock.earnings, headers='keys', tablefmt='pretty', showindex=True))

def display_quarterly_earnings(stock):
    print("### Quarterly Earnings")
    print(tabulate(stock.quarterly_earnings, headers='keys', tablefmt='pretty', showindex=True))

def display_sustainability(stock):
    print("### Sustainability")
    print(tabulate(stock.sustainability, headers='keys', tablefmt='pretty', showindex=True))

def display_recommendations(stock):
    print("### Analysts Recommendations")
    print(tabulate(stock.recommendations, headers='keys', tablefmt='pretty', showindex=True))

def display_calendar(stock):
    print("### Calendar (Next Events, Earnings, etc.)")
    print(tabulate(stock.calendar, headers='keys', tablefmt='pretty', showindex=True))

def display_isin(stock):
    print("### ISIN (International Securities Identification Number)")
    print(stock.isin)

def display_options(stock):
    print("### Options Expirations")
    print(stock.options)

def display_news(stock):
    print("### Latest News")
    news_data = stock.news
    
    if not news_data:
        print("No news available.")
        return
    
    for article in news_data:
        content = article.get('content', {})
        title = content.get('title', 'No Title')
        summary = content.get('summary', 'No summary available.')
        pub_date = content.get('pubDate', 'No date available.')
        link = content.get('clickThroughUrl', {}).get('url', '#')
        
        # Safely retrieve the thumbnail
        thumbnail = None
        if 'thumbnail' in content and content['thumbnail'] is not None:
            thumbnail = content['thumbnail'].get('originalUrl', None)
        
        print(f"**{title}**")
        print(f"Published: {pub_date}")
        print(f"Summary: {summary}")
        
        # Display the thumbnail if available
        if thumbnail:
            print(f"Thumbnail: {thumbnail}")
        
        # Provide a clickable link to the full article
        print(f"Read full article: {link}")
        print("\n---\n")

# -------------------------------------------------------------------------------------
# Step 3: Main Logic
# -------------------------------------------------------------------------------------
def main():
    ticker = input('Enter Stock Ticker (e.g. D05.SI, AAPL): ')

    if ticker:
        stock = get_stock_data(ticker)

        # Display the candlestick chart
        plot_candlestick_chart(stock)

        # Display stock information
        display_stock_info(stock)
        display_stock_actions(stock)
        display_stock_financials(stock)
        display_stock_quarterly_financials(stock)
        display_major_holders(stock)
        display_institutional_holders(stock)
        display_balance_sheet(stock)
        display_quarterly_balance_sheet(stock)
        display_cashflow(stock)
        display_quarterly_cashflow(stock)
        display_earnings(stock)
        display_quarterly_earnings(stock)
        display_sustainability(stock)
        display_recommendations(stock)
        display_calendar(stock)
        display_isin(stock)
        display_options(stock)
        display_news(stock)

if __name__ == '__main__':
    main()


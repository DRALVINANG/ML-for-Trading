import yfinance as yf
import pandas as pd
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Comprehensive pandas display configuration
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

#---------------------------------------------------------------------------------
# Part 1: Stock Info
#---------------------------------------------------------------------------------
ticker = 'D05.SI'
stock = yf.Ticker(ticker)

print("### Stock Information")
try:
    info = stock.info
    df1 = pd.DataFrame(info.items(), columns=["Key", "Value"])
    print(df1)
except Exception as e:
    print(f"Error retrieving stock info: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 2: Stock Actions (Dividends & Splits)
#---------------------------------------------------------------------------------
print("### Stock Actions (Dividends & Splits)")
try:
    actions_data = stock.actions
    if actions_data.empty:
        print("No actions data available")
    else:
        print(actions_data)
except Exception as e:
    print(f"Error retrieving actions: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 3: Stock Financials
#---------------------------------------------------------------------------------
print("### Stock Financials")
try:
    financials_data = stock.financials
    if financials_data.empty:
        print("No financials data available")
    else:
        print(financials_data)
except Exception as e:
    print(f"Error retrieving financials: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 4: Stock Quarterly Financials
#---------------------------------------------------------------------------------
print("### Stock Quarterly Financials")
try:
    quarterly_financials_data = stock.quarterly_financials
    if quarterly_financials_data.empty:
        print("No quarterly financials data available")
    else:
        print(quarterly_financials_data)
except Exception as e:
    print(f"Error retrieving quarterly financials: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 5: Major Holders
#---------------------------------------------------------------------------------
print("### Major Holders")
try:
    major_holders_data = stock.major_holders
    if major_holders_data is None or major_holders_data.empty:
        print("No major holders data available")
    else:
        print(major_holders_data)
except Exception as e:
    print(f"Error retrieving major holders: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 6: Institutional Holders
#---------------------------------------------------------------------------------
print("### Institutional Holders")
try:
    institutional_holders_data = stock.institutional_holders
    if institutional_holders_data is None or institutional_holders_data.empty:
        print("No institutional holders data available")
    else:
        print(institutional_holders_data)
except Exception as e:
    print(f"Error retrieving institutional holders: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 7: Balance Sheet
#---------------------------------------------------------------------------------
print("### Balance Sheet")
try:
    balance_sheet_data = stock.balance_sheet
    if balance_sheet_data.empty:
        print("No balance sheet data available")
    else:
        print(balance_sheet_data)
except Exception as e:
    print(f"Error retrieving balance sheet: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 8: Quarterly Balance Sheet
#---------------------------------------------------------------------------------
print("### Quarterly Balance Sheet")
try:
    quarterly_balance_sheet_data = stock.quarterly_balance_sheet
    if quarterly_balance_sheet_data.empty:
        print("No quarterly balance sheet data available")
    else:
        print(quarterly_balance_sheet_data)
except Exception as e:
    print(f"Error retrieving quarterly balance sheet: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 9: Cashflow
#---------------------------------------------------------------------------------
print("### Cashflow")
try:
    cashflow_data = stock.cashflow
    if cashflow_data.empty:
        print("No cashflow data available")
    else:
        print(cashflow_data)
except Exception as e:
    print(f"Error retrieving cashflow: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 10: Quarterly Cashflow
#---------------------------------------------------------------------------------
print("### Quarterly Cashflow")
try:
    quarterly_cashflow_data = stock.quarterly_cashflow
    if quarterly_cashflow_data.empty:
        print("No quarterly cashflow data available")
    else:
        print(quarterly_cashflow_data)
except Exception as e:
    print(f"Error retrieving quarterly cashflow: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 11: Earnings (using income_stmt instead)
#---------------------------------------------------------------------------------
print("### Earnings (Income Statement)")
try:
    earnings_data = stock.income_stmt
    if earnings_data.empty:
        print("No earnings data available")
    else:
        print(earnings_data)
except Exception as e:
    print(f"Error retrieving earnings data: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 12: Quarterly Earnings
#---------------------------------------------------------------------------------
print("### Quarterly Earnings")
try:
    quarterly_earnings_data = stock.quarterly_income_stmt
    if quarterly_earnings_data is None or quarterly_earnings_data.empty:
        print("No quarterly earnings data available")
    else:
        print(quarterly_earnings_data)
except Exception as e:
    print(f"Error retrieving quarterly earnings data: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 13: Sustainability
#---------------------------------------------------------------------------------
print("### Sustainability")
try:
    sustainability_data = stock.sustainability
    if sustainability_data is None or sustainability_data.empty:
        print("No sustainability data available")
    else:
        print(sustainability_data)
except Exception as e:
    print(f"Error retrieving sustainability data: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 14: Analyst Recommendations
#---------------------------------------------------------------------------------
print("### Analyst Recommendations")
try:
    recommendations_data = stock.recommendations
    if recommendations_data is None or recommendations_data.empty:
        print("No analyst recommendations data available")
    else:
        print(recommendations_data)
except Exception as e:
    print(f"Error retrieving analyst recommendations: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 15: Calendar (Next Events, Earnings, etc.)
#---------------------------------------------------------------------------------
print("### Calendar (Next Events, Earnings, etc.)")
try:
    calendar_data = stock.calendar
    if calendar_data is None or calendar_data.empty:
        print("No calendar data available")
    else:
        print(calendar_data)
except Exception as e:
    print(f"Error retrieving calendar data: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 16: ISIN
#---------------------------------------------------------------------------------
print("### ISIN (International Securities Identification Number)")
try:
    isin_data = stock.isin
    if isin_data:
        print(isin_data)
    else:
        print("No ISIN data available")
except Exception as e:
    print(f"Error retrieving ISIN: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 17: Options Expirations
#---------------------------------------------------------------------------------
print("### Options Expirations")
try:
    options_data = stock.options
    if options_data:
        print(options_data)
    else:
        print("No options data available")
except Exception as e:
    print(f"Error retrieving options data: {e}")
input("\nPress Enter to continue to next section...")
print("\n" + "="*80 + "\n")

#---------------------------------------------------------------------------------
# Part 18: News
#---------------------------------------------------------------------------------
print("### Latest News")
try:
    news_data = stock.news

    if not news_data:
        print("No news available.")
    else:
        for i, article in enumerate(news_data, 1):
            # Safely get content with multiple fallbacks
            content = article.get('content', {}) if article else {}
            if content is None:
                content = {}
            
            title = content.get('title', article.get('title', 'No Title'))
            summary = content.get('summary', article.get('summary', 'No summary available.'))
            pub_date = content.get('pubDate', article.get('providerPublishTime', 'No date available.'))
            
            # Safely get link
            link = '#'
            click_through = content.get('clickThroughUrl', {})
            if click_through and isinstance(click_through, dict):
                link = click_through.get('url', '#')
            elif article.get('link'):
                link = article.get('link', '#')
            
            print(f"News {i}: {title}")
            print(f"Published: {pub_date}")
            if summary and len(summary) > 200:
                print(f"Summary: {summary[:200]}...")
            else:
                print(f"Summary: {summary}")
            print(f"Link: {link}")
            print("-" * 50)
            
except Exception as e:
    print(f"Error retrieving news: {e}")

input("\nPress Enter to finish...")

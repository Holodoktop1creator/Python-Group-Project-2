import os
import requests
import time
import csv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

API_URL = 'https://api.nytimes.com/svc/books/v3/lists/overview.json'

START_PUBLISHED_DATE = '2025-09-11'

PREVIOUS_PUBLISHED_DATE = '2025-01-01'


OUT_CSV = 'book.csv'

FIELDS = ["author", "created_date", "description", "price", "publisher", "title", "rank"]


def fetch_week(api_key, published_date):
    params = {'api-key': api_key}
    if published_date:
        params['published_date'] = published_date
    response = requests.get(API_URL, params=params)
    logging.info('GET url=%s, published-date=%s', API_URL, published_date)
    response.raise_for_status()
    return response.json()



def parse_json(data):
    answer = []

    results = data.get("results", {})
    lists = results.get('lists', [])
    for i in lists:
        books = i.get('books', [])
        for j in books:
            book = {}
            for l in FIELDS:
                book[l] = j.get(l, '')
            answer.append(book)
    previous_published_date = results.get('previous_published_date')   
    logging.info('Got %s books, preview_published_date = %s', len(answer), previous_published_date)      
    return answer, previous_published_date
    

                    




def main():
    api_key = os.getenv('API_KEY')
    if not api_key:
        logging.error('API_KEY not found')
        raise Exception('API_KEY не установлено')
    rows = []
    logging.info('Start parsing')
    previous_published_date = START_PUBLISHED_DATE
    while previous_published_date and previous_published_date >= PREVIOUS_PUBLISHED_DATE:
        data = fetch_week(api_key, previous_published_date)

        answer, previous_published_date = parse_json(data)
        rows.extend(answer)
        logging.info('Date received, sleep for 12 s')
        time.sleep(12)
    logging.info('Start to save in file = %s', OUT_CSV)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    logging.info('Programm finished')

if __name__ == '__main__':
    main()
    
        

    


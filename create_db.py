import sqlite3
import json
import time

test_initial_file = '/home/Arsuh/Desktop/RC_2018-09.txt'
test_path = './Reddit_db.db'

initial_file = '/run/media/Arsuh/DATA/Reddit_db/RC_2018-09.txt'
path = '/run/media/Arsuh/DATA/Reddit_db/Reddit_db.db'

conn = sqlite3.connect(path)
c = conn.cursor()
transaction = []
cleanup = 1000000


def add_transaction(query):
    global transaction
    transaction.append(query)
    if len(transaction) > 2000:
        c.execute('BEGIN TRANSACTION')
        for q in transaction:
            try:
                c.execute(q)
            except:
                pass
        conn.commit()
        transaction = []


def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS mainTable (parent_id TEXT, comment_id TEXT UNIQUE, parent TEXT, comment TEXT, score INT, subreddit TEXT)")


def isvalid(comment):
    if len(comment.split(' ')) > 500 or len(comment) < 1:
        return False
    if len(comment) > 1500:
        return False
    if comment[:6] == 'https:' or comment[1:7] == 'https:' or comment[:5] == 'http:' or comment[1:6] == 'http:' or comment[:4] == 'www.' or comment[1:5] == 'www.':
        return False
    if comment == '[deleted]' or comment == '[removed]':
        return False
    return True


def find_parent(pid):
    query = 'SELECT comment FROM mainTable WHERE comment_id="{}" LIMIT 1'.format(
        pid)
    c.execute(query)
    result = c.fetchone()
    if result != None:
        return result[0]  # <-- try/except
    return '#NaN#'


def find_existing_score(pid):
    query = 'SELECT score FROM mainTable WHERE parent_id="{}" LIMIT 1'.format(
        pid)
    c.execute(query)
    result = c.fetchone()
    if result != None:
        return result[0]
    return '#NaN#'


def update_entry(parent_id, comment_id, parent, comment, score, subreddit):
    query = 'UPDATE mainTable SET parent_id="{}", comment_id="{}", parent="{}", comment="{}", score={}, subreddit="{}" WHERE parent_id="{}"'.format(
        parent_id, comment_id, parent, comment, score, subreddit, parent_id)
    add_transaction(query)
    # c.execute(query)
    # conn.commit()


def populate(parent_id, comment_id, parent, comment, score, subreddit):
    query = 'INSERT INTO mainTable (parent_id, comment_id, parent, comment, score, subreddit) VALUES("{}", "{}", "{}", "{}", {}, "{}")'.format(
        parent_id, comment_id, parent, comment, score, subreddit)
    add_transaction(query)


if __name__ == '__main__':
    start = time.time()
    create_table()
    row_counter = 0
    paired_rows = 0
    ckpt_time = time.time()

    with open(initial_file, buffering=2000) as f:
        for row in f:
            row_counter += 1

            try:
                row = json.loads(row)
                score = row['score']
                subreddit = row['subreddit']
                if score < 0 or subreddit == 'MemeEconomy':
                    row_counter -= 1
                    continue

                comment = row['body'].replace('\r', '\n').replace("'", '"')
                if isvalid(comment) != True:
                    row_counter -= 1
                    continue

                parent_id = row['parent_id'][3:]
                comment_id = row['id']
                parent = find_parent(parent_id)

                if parent != '#NaN#':
                    paired_rows += 1

                populate(parent_id, comment_id, parent,
                         comment, score, subreddit)

            except Exception as e:
                print(str(e))

            if row_counter % 200000 == 0:
                current_time = time.time()
                print('Rows processed: {} | Paired rows: {} | Time: {:.2f} mins'.format(
                    row_counter, paired_rows, (current_time-ckpt_time)/60))
                ckpt_time = current_time

            if row_counter % cleanup == 0:
                print('   >>> Deleting rows...')
                c.execute("DELETE FROM mainTable WHERE parent == '#NaN#'")
                conn.commit()

    print('   >>> Deleting rows...')
    c.execute("DELETE FROM mainTable WHERE parent == '#NaN#'")
    conn.commit()

    c.close()
    conn.close()
    print('Time taken: {:.2f} mins / {:.2f} hrs'.format(
        (time.time()-start)/60, (time.time()-start)/3600))

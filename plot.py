import matplotlib.pyplot as plt

paths = ['./checkpoints/checkpoints-125ep-3/',
         './checkpoints/checkpoints-final-2/',
         './']

def load_data():
    sessions = []
    for path in paths:
        try:
            session = []
            with open(path + 'plot.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    session.append(float(line))
                sessions.append(session)
        except Exception as e:
            print(e)
    return sessions

'''
def load_data():
    sessions = []
    session = []
    with open('./plot.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            session.append(float(line))
        sessions.append(session)

    return sessions
'''
if __name__ == '__main__':
    sessions = load_data()

    fig = plt.figure()
    fig.suptitle('Training Sessions')
    for session in sessions:
        plt.plot(session)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['sess'+str(i+1) for i in range(len(sessions))], loc='upper right')
    plt.show()

import fcntl, os

class Job:
    lockFileName = 'gpu.lock'
    
    def __init__(self, action):
        self.action = action

    def run(self):
        """
        THIS IS BLOCKING! CALL IT AT LAST!
        """
        with open(Job.lockFileName, 'w') as f:
            rv = fcntl.lockf(f.fileno(), fcntl.LOCK_EX)
            print("job {} is running.".format(os.getpid()))
            f.write(str(os.getpid()) + '\n')
            self.action()
            fcntl.lockf(f.fileno(), fcntl.LOCK_UN)

if __name__ == '__main__':
    pass
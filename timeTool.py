import time
class timeTool:
    def start(self):
        self.tic = time.time()
    def end(self):
        self.toc = time.time()
        usetime = self.toc - self.tic
        mins = int(usetime/60)
        secds = usetime%60
        print(f"completed! usetime = {mins}min,{secds:.2f}s")
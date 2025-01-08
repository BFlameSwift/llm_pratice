import os


class myTokenizer:
    def __init__(self, text_file):
        self.replacement = {}
        self.vocab_size = 1024
        self.merge_num = 1024 - 256

        with open(text_file, "r") as f:
            lines = f.readlines()
            self.raw_text = "\n".join(lines)

        self.ids = list(self.raw_text.encode("utf-8"))
        self.original_ids = self.ids.copy()

    def get_stats(self, ids, counter={}):
        ret = None
        if len(ids) < 2:
            return None

        for pair in zip(ids, ids[1:]):
            counter[pair] = counter.get(pair, 0) + 1

        ret = max(counter, key=counter.get)

        return ret

    def merge(self, ids, pair, new_id):
        new_ids = []
        i = 0

        while i < len(ids):
            if i < len(ids) - 1 and pair[0] == ids[i] and pair[1] == ids[i + 1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        self.replacement[new_id] = pair
        return new_ids

    def encode(self):
        while self.merge_num > 0:
            pair = self.get_stats(self.ids)

            self.ids = self.merge(self.ids, pair, self.vocab_size - self.merge_num)
            self.merge_num -= 1

    def decode(self):
        for i in range(self.vocab_size, 255, -1):
            new_ids = []
            for id in self.ids:
                if id == i:
                    new_ids.append(self.replacement[i][0])
                    new_ids.append(self.replacement[i][1])
                else:
                    new_ids.append(id)
            self.ids = new_ids
        # return ids

    # print("1",end='\t


if __name__ == "__main__":
    tokenizer = myTokenizer('./manual.txt')
    print(len(tokenizer.ids))
    tokenizer.encode()
    print(len(tokenizer.ids))
    tokenizer.decode()
    print(len(tokenizer.ids))
    print(tokenizer.ids == tokenizer.original_ids)

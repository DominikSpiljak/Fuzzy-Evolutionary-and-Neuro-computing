class Individual:

    def __init__(self, value, fitness_func):
        self.value = value
        self.fitness = fitness_func(*value)

    @staticmethod
    def encode_value(value):
        # Svaki parametar je reprezentiran sa 8 brojeva [0 - 1]
        encoded_value = []
        for num in value:
            num = num + 4
            encoded_part = num / 8
            encoded_value.extend([encoded_part] * 8)
        return encoded_value

    @staticmethod
    def decode_value(value):
        decoded_value = []
        start = 0
        end = 8
        while start < 40:
            decoded_value.append(sum(value[start:end]) - 4)
            start = end
            end += 8
        return decoded_value

    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return "Betas = {}, Fitness = {}".format(self.value, self.fitness)

"""
Smoker:     No      -> 0
            Casual  -> 0.5
            Yes     -> 1

Carbonated beverages    |   Never           -> 0
Tomatoes                |   Almost Never    -> 1
Coffee                  |   Sometimes       -> 2
Chocolate               |   Almost Always   -> 3
Soft cheese             |   Always          -> 4
Citrus fruits           |

Alcohol Consumption:    Non-Drinker         -> 0
                        Casual Drinker      -> 1
                        Habitual Drinker    -> 2

"""

import re

class preprocessor:
    def __init__(self, filename):
        self.avg_citrus_weight = 130    # In Grams
        self.cat_options = ['never', 'almost never', 'sometimes', 'almost always', 'always']
        self.alch_consum = ['nondrinker', 'casual drinker', 'habitual drinker']
        self.info_dict = {}
        with open(filename, 'r') as f:
            lines = f.readlines()
            condition = lambda line: re.sub(r'[\t]', ':', line) if ':' not in line else line
            cleaned_lines = [condition(line) for line in lines]
            cleaned_lines = [re.sub(r'[\n\t]', '', s) for s in cleaned_lines]
            cleaned_lines= [item for item in cleaned_lines if item != ':']
            numerical_fields = ['Age', 'Voice Handicap Index (VHI) Score', 'Reflux Symptom Index (RSI) Score', 'Number of cigarettes smoked per day', 'Number of glasses containing alcoholic beverage drinked in a day', 'Amount of water\'s litres drink every day', 'Amount of glasses drinked in a day', 'Number of cups of coffee drinked in a day', 'Gramme of chocolate eaten in  a day', 'Gramme of soft cheese eaten in a day', 'Number of citrus fruits eaten in a day']
            general_categorical = ['Carbonated beverages', 'Tomatoes', 'Coffee', 'Chocolate', 'Soft cheese', 'Citrus fruits']
            for pair in cleaned_lines:
                li = pair.split(':')
                if li[0] == 'Smoker':
                    smoker_map = lambda x: 0 if x.lower() == 'no' else (1 if x.lower() == 'yes' else 0.5)
                    li[1] = smoker_map(li[1])
                elif li[0] in numerical_fields:
                    li[1] = self.numerical_map(li[1], li[0])
                elif li[0] in general_categorical:
                    li[1] = self.cat_options.index(li[1])
                elif li[0] == 'Alcohol consumption':
                    li[1] = self.alch_consum_map(li[1])

                self.info_dict[li[0]] = li[1]

    def numerical_map(self, input_string, data_field):
        # replace NU with Null
        # fix decimals that are written using a comma instead of a fullstop
        # Some numbers have been written as fraction eg. 1/2, so these need to be converted to decimals
        # For the amount of soft cheese eaten per day, they sometimes write a range of values using a forward slash. eg 100/150 means a range of 100 to 150
        # remove 'for month', 'for week' and adjust value accordingly (Don't forget month has sometimes been spelt as mounth)
        # 'for month' is sometimes written as '/ month' so this also requires the value to be adjusted accordingly

        if input_string.lower() == 'nu':
            return None
        
        input_string = re.sub(',', '.', input_string)

        val = re.findall(r'[0-9]+\/[0-9]+|[0-9]+[\-][0-9]+|[0-9]+.[0-9]+|[0-9]+', input_string)[0]
        if '/' in val:
            if data_field == 'Gramme of soft cheese eaten in a day':
                tmp_li = val.split('/')
                val = (float(tmp_li[0]) + float(tmp_li[1])) / 2
            else:
                val = eval(val)
        elif '-' in val:
            tmp_li = val.split('-')
            val = (float(tmp_li[0]) + float(tmp_li[1])) / 2
        else:
            val = float(val)

        if 'for week' in input_string or 'per week' in input_string:
            val /= 7
        elif 'for month' in input_string or 'for mounth' in input_string or '/ month' in input_string:
            val /= 28

        if data_field == 'Number of citrus fruits eaten in a day' and 'gramme' in input_string:
            val /= self.avg_citrus_weight

        return val

    def alch_consum_map(self, input_val):
        if self.alch_consum[0] in input_val:
            return 0
        elif self.alch_consum[1] in input_val:
            return 1
        elif self.alch_consum[2] in input_val:
            return 2
        else:
            return None

    def get_info(self):
        return self.info_dict
         

if __name__ == "__main__":
    prep = preprocessor("data/voice002-info.txt")
    print(prep.get_info())
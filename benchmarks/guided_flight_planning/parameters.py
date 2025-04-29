# Environment Parameters
SIMULATE_FLYING = True  # Removing this parameter makes the model rely only on geographical coordinates
INCLUDE_ATMOS = True  # Include atmospheric conditions about wind (magnitude and direction)
INCLUDE_T = True  # Include atmospheric conditions about temperature
PERF_MODEL = "PS"  # PS or openap

# Agent Parameters
HIDDEN_SIZE = 64  # Size of the hidden layer of the Agent's neural network. 64 is the default value.
CLIP_RANGE = 0.2  # Clipping range for the PPO algorithm. 0.2 is the default value.
PPO_LR = 0.00005  # Learning rate for the PPO algorithm. 0.00005 is the default value.

# Training Parameters
TRAIN_ITERATIONS = 16000  # Number of iterations for training the Agent. 16000 is the default value.
ITERATIONS_PER_SAVE = TRAIN_ITERATIONS / 1000  # Number of iterations between each model save.
PENALTY = 0.05  # Fixed penalty for the Agent's reward function. 0.05 is the default value.
REWARD_EXP = 2  # Importance factor for the Agent's reward function. >1 gives more importance to the direction of the route, <1 gives more importance to the fuel consumption.
LOAD_MODEL = True  # True activates the testing mode, False activates the training mode.
USE_LOWEST_LOSS = True  # If True, the model with the lowest loss is saved. If False, the last trained model is saved.
EXTRA_END_REWARD = True  # If True, the Agent receives an additional reward for reaching the target.

# RL Domain Parameters
FWD_POINTS = 41
LAT_POINTS = 11
VERTICAL_POINTS = 6

MIN_LENGTH = 4  # Used for random trips in training
MAX_LENGTH = 8
MIN_CUT = 0.33  # Minimum portion of the graph to be considered (expressed as a percentage)
MAX_CUT = 0.5  # Maximum portion of the graph to be considered (expressed as a percentage)

# Generic Parameters
MODEL_PATH = "Models/"
NPZ_PATH = None

# Testing Parameters
FIXED_TESTS = {
    "From Airport": ["Frankfurt", "London Heathrow", "Madrid Barajas", "Rome Fiumicino", "Munich",
                     "Barcelona El Prat", "Stockholm Arlanda", "Prague", "Zurich", "Brussels",
                     "Bucharest", "Zagreb", "Athens", "Lisbon Portela", "Helsinki Vantaa", "Sofia",
                     "Copenhagen", "Amsterdam Schiphol", "London Gatwick", "Oslo Gardermoen",
                     "Vienna Schwechat", "Budapest", "Paris Orly", "Venice Marco Polo", "Milan Linate",
                     "Hamburg", "Toulouse Blagnac", "Athens", "Lisbon Portela", "Berlin Brandenburg",
                     "Prague", "Dublin"],
    "From IATA Code": ["FRA", "LHR", "MAD", "FCO", "MUC",
                       "BCN", "ARN", "PRG", "ZRH", "BRU",
                       "OTP", "ZAG", "ATH", "LIS", "HEL",
                       "SOF", "CPH", "AMS", "LGW", "OSL",
                       "VIE", "BUD", "ORY", "VCE", "LIN",
                       "HAM", "TLS", "ATH", "LIS", "BER",
                       "PRG", "DUB"],
    "From Latitude": [50.0379, 51.4700, 40.4719, 41.8003, 48.3538,
                      41.2971, 59.6519, 50.1008, 47.4647, 50.9014,
                      44.5711, 45.7429, 37.9364, 38.7742, 60.3172,
                      42.6965, 55.6181, 52.3086, 51.1537, 60.2028,
                      48.1103, 47.4396, 48.7253, 45.5053, 45.4451,
                      53.6304, 43.6293, 37.9364, 38.7742, 52.3667,
                      50.1008, 53.4213],
    "From Longitude": [8.5622, -0.4543, -3.5626, 12.2389, 11.7861,
                       2.0785, 17.9186, 14.2632, 8.5492, 4.4844,
                       26.085, 16.0688, 23.9445, -9.1342, 24.9633,
                       23.4114, 12.656, 4.7639, -0.1821, 11.0839,
                       16.5697, 19.261, 2.3594, 12.3519, 9.2767,
                       9.9882, 1.3638, 23.9445, -9.1342, 13.5033,
                       14.2632, -6.2701],
    "From Altitude": [364, 83, 609, 13, 520, 12, 137, 400, 408, 184,
                      314, 353, 308, 374, 179, 699, 17, -11, 202, 681,
                      600, 495, 291, 7, 353, 53, 499, 308, 374, 157,
                      400, 242],
    "To Airport": ['Paris Charles de Gaulle', 'Amsterdam Schiphol','Lisbon Portela', 'Vienna Schwechat', 'Berlin Brandenburg',
                   'Brussels', 'Copenhagen', 'Warsaw Chopin', 'Budapest', 'Vienna Schwechat',
                   'Vienna Schwechat', 'Frankfurt', 'Milan Malpensa', 'Dublin', 'Munich',
                   'Athens', 'Amsterdam Schiphol', 'Paris Charles de Gaulle', 'Rome Fiumicino', 'Stockholm Arlanda',
                   'Budapest', 'Bucharest', 'Madrid Barajas', 'Frankfurt', 'Brussels',
                   'Berlin Brandenburg', 'Lisbon Portela', 'Munich', 'Paris Charles de Gaulle', 'Amsterdam Schiphol',
                   'Munich', 'London Heathrow'],
    "To IATA Code": ['CDG', 'AMS', 'LIS', 'VIE', 'BER', 'BRU',
                     'CPH', 'WAW', 'BUD', 'VIE', 'VIE', 'FRA',
                     'MXP', 'DUB', 'MUC', 'ATH', 'AMS', 'CDG',
                     'FCO', 'ARN', 'BUD', 'OTP', 'MAD', 'FRA',
                     'BRU', 'BER', 'LIS', 'MUC', 'CDG', 'AMS',
                     'MUC', 'LHR'],
    "To Latitude": [49.0097, 52.3086, 38.7742, 48.1103, 52.3667,
                    50.9014, 55.6181, 52.1657, 47.4396, 48.1103,
                    48.1103, 50.0379, 45.6301, 53.4213, 48.3538,
                    37.9364, 52.3086, 49.0097, 41.8003, 59.6519,
                    47.4396, 44.5711, 40.4719, 50.0379, 50.9014,
                    52.3667, 38.7742, 48.3538, 49.0097, 52.3086,
                    48.3538, 51.47],
    "To Longitude": [2.5479, 4.7639, -9.1342, 16.5697, 13.5033,
                     4.4844, 12.656, 20.9671, 19.261, 16.5697,
                     16.5697, 8.5622, 8.7231, -6.2701, 11.7861,
                     23.9445, 4.7639, 2.5479, 12.2389, 17.9186,
                     19.261, 26.085, -3.5626, 8.5622, 4.4844,
                     13.5033, -9.1342, 11.7861, 2.5479, 4.7639,
                     11.7861, -0.4543],
    "To Altitude": [392, -11, 374, 600, 157,
                    184, 17, 362, 495, 600,
                    600, 364, 768, 242, 520,
                    308, -11, 392, 13, 137,
                    495, 314, 609, 364, 184,
                    157, 374, 520, 392, -11,
                    520, 83]
}  # Fixed routes for testing the Agent (used in fixed_test.py)
WIND_DS = [None]
COMPUTED_NOISE = {}

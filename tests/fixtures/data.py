import numpy as np

# np.random.rand(100)
f1 = np.array([0.10237979, 0.55533541, 0.78195433, 0.97841412, 0.83085375,
               0.7455798, 0.49168234, 0.75093463, 0.17067203, 0.78884921,
               0.78959655, 0.77734149, 0.79967087, 0.95010057, 0.90290046,
               0.53899196, 0.52015989, 0.05845868, 0.01695446, 0.01471052,
               0.28130947, 0.87701811, 0.43277188, 0.86034914, 0.82725378,
               0.7067565, 0.02221204, 0.53623852, 0.23087543, 0.43422264,
               0.89022442, 0.90081728, 0.3133706, 0.5354169, 0.02442898,
               0.16752663, 0.55306834, 0.39346516, 0.13016247, 0.43174516,
               0.5229127, 0.12961708, 0.68634881, 0.62692116, 0.15089027,
               0.7696994, 0.76759285, 0.45542104, 0.4995261, 0.96357856,
               0.69033989, 0.36240883, 0.75573588, 0.02828793, 0.69871438,
               0.46191817, 0.11499773, 0.89409187, 0.78956961, 0.08228614,
               0.4270966, 0.03859564, 0.55172972, 0.50639862, 0.64629711,
               0.98081932, 0.31034672, 0.22668293, 0.35562919, 0.31283607,
               0.56685852, 0.27430796, 0.63206713, 0.40590245, 0.66063327,
               0.46816459, 0.62981937, 0.16869701, 0.65446764, 0.09366221,
               0.82613006, 0.96993874, 0.81727168, 0.0040796, 0.09578179,
               0.67860984, 0.43566862, 0.39898807, 0.71003928, 0.90102993,
               0.93841659, 0.30903815, 0.63057767, 0.92023451, 0.94742813,
               0.50406069, 0.74908966, 0.30283682, 0.97144493, 0.00718653])

# np.random.rand(100)
f2 = np.array([0.8777517, 0.15629693, 0.89371361, 0.1308805, 0.89113409,
               0.31010849, 0.62753642, 0.33815208, 0.98519193, 0.19204633,
               0.47216134, 0.62400463, 0.87075792, 0.25295231, 0.89039409,
               0.82243837, 0.90666689, 0.9213208, 0.55456055, 0.9236469,
               0.00294893, 0.99692265, 0.09389125, 0.5867195, 0.69832993,
               0.85477242, 0.43610932, 0.10300754, 0.22841349, 0.2224899,
               0.00479793, 0.8142253, 0.22528547, 0.74534865, 0.10688432,
               0.74130184, 0.04384125, 0.20526685, 0.89518007, 0.37365158,
               0.49485723, 0.42443405, 0.40832973, 0.87919414, 0.7057943,
               0.75230302, 0.17673952, 0.71054438, 0.42396852, 0.75234724,
               0.93645999, 0.27291753, 0.75147405, 0.0617426, 0.48151646,
               0.36198956, 0.038121, 0.05681121, 0.25913534, 0.78834303,
               0.01115275, 0.48459944, 0.91816375, 0.46291374, 0.26549133,
               0.95714558, 0.30439443, 0.46311506, 0.41903678, 0.67452356,
               0.58937203, 0.97467556, 0.21936688, 0.33080927, 0.8472346,
               0.14323858, 0.29807776, 0.77213025, 0.72966189, 0.60364796,
               0.66352878, 0.36535455, 0.99900726, 0.73277024, 0.44223856,
               0.43351931, 0.39707756, 0.38925028, 0.28672124, 0.41595846,
               0.80155977, 0.24705746, 0.39228539, 0.60629254, 0.15213419,
               0.09662356, 0.62639217, 0.6954235, 0.1363836, 0.55174753])

# np.random.rand(100)
f3 = np.array([0.60847141, 0.21432351, 0.33349512, 0.30889298, 0.13009325,
               0.13662646, 0.00695053, 0.75397825, 0.16614486, 0.0851955,
               0.50508386, 0.75975802, 0.6645811, 0.23836495, 0.09804966,
               0.00294266, 0.39413401, 0.25096607, 0.82676041, 0.24848367,
               0.35556693, 0.69488518, 0.31625948, 0.15742354, 0.74727023,
               0.96715756, 0.37694339, 0.8610378, 0.12038739, 0.57964345,
               0.60514525, 0.57811029, 0.70172484, 0.10421092, 0.70595904,
               0.96709041, 0.12464899, 0.7592479, 0.81997126, 0.91397242,
               0.20514423, 0.24652297, 0.7576649, 0.44402663, 0.74847175,
               0.08312819, 0.51352849, 0.10690292, 0.31923125, 0.26378033,
               0.55611577, 0.36883027, 0.11776463, 0.23752719, 0.95285463,
               0.6584294, 0.07778219, 0.6014062, 0.48531944, 0.16785589,
               0.41781332, 0.62092328, 0.90764333, 0.20540448, 0.68773383,
               0.27106715, 0.1282443, 0.1559362, 0.62639941, 0.44188011,
               0.25267947, 0.053747, 0.64548594, 0.60117854, 0.57953808,
               0.38645014, 0.10492664, 0.21617922, 0.35081033, 0.64989639,
               0.09376668, 0.86942725, 0.31075929, 0.83919955, 0.2815903,
               0.75502223, 0.30992443, 0.8515371, 0.01852557, 0.05983628,
               0.73731295, 0.31043444, 0.97974229, 0.10572803, 0.84496669,
               0.50049798, 0.21981922, 0.64462382, 0.26108781, 0.7181113])

# np.random.rand(100)
y = np.array([0.89539184, 0.07019512, 0.35920388, 0.16511609, 0.32126577,
              0.40867521, 0.18649066, 0.9291251, 0.85796662, 0.43139191,
              0.10788777, 0.57374487, 0.97005752, 0.06610726, 0.67105467,
              0.44713098, 0.39346228, 0.50867755, 0.21352044, 0.79479963,
              0.24337954, 0.54678784, 0.06586959, 0.91494179, 0.11389051,
              0.04281806, 0.54305643, 0.24617006, 0.58984712, 0.53249882,
              0.77950262, 0.82571049, 0.11930034, 0.68248065, 0.40339771,
              0.12965398, 0.62971889, 0.35325503, 0.71314607, 0.81567706,
              0.49208879, 0.28296996, 0.52472319, 0.59129255, 0.79982189,
              0.66423439, 0.56383144, 0.92928899, 0.26893795, 0.41810213,
              0.94829342, 0.1537145, 0.19159626, 0.35668969, 0.00365191,
              0.93902533, 0.47790748, 0.7741865, 0.4716689, 0.82962576,
              0.70120752, 0.22107045, 0.34414003, 0.994593, 0.16381657,
              0.52696272, 0.79831738, 0.56694781, 0.48393758, 0.54868185,
              0.76319717, 0.41323983, 0.07839293, 0.57203942, 0.09683323,
              0.0315013, 0.68507295, 0.08016741, 0.2310444, 0.15004811,
              0.46720155, 0.67467499, 0.1457815, 0.68632132, 0.71329273,
              0.13685809, 0.16994026, 0.38038981, 0.90059071, 0.04801532,
              0.01357006, 0.44065687, 0.57318312, 0.40958057, 0.72930387,
              0.38554057, 0.0892655, 0.59395324, 0.9060076, 0.60680206])

# [np.ones(100)]
H0 = np.array([[1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.],
               [1.]])

# [np.ones(100), np.random.rand(100)]
H1 = np.array([[1., 0.30383193],
               [1., 0.72439984],
               [1., 0.66692085],
               [1., 0.23974312],
               [1., 0.18991335],
               [1., 0.47496919],
               [1., 0.9400303],
               [1., 0.8037643],
               [1., 0.60785564],
               [1., 0.92039894],
               [1., 0.96280047],
               [1., 0.86551393],
               [1., 0.38964882],
               [1., 0.24947293],
               [1., 0.49001349],
               [1., 0.310824],
               [1., 0.5526414],
               [1., 0.94906822],
               [1., 0.09367857],
               [1., 0.2122386],
               [1., 0.71772278],
               [1., 0.02824406],
               [1., 0.47135494],
               [1., 0.76336479],
               [1., 0.7849169],
               [1., 0.5252335],
               [1., 0.01282522],
               [1., 0.66583501],
               [1., 0.39626153],
               [1., 0.61223058],
               [1., 0.74855368],
               [1., 0.7478602],
               [1., 0.80946826],
               [1., 0.48326257],
               [1., 0.7338476],
               [1., 0.71041196],
               [1., 0.23770007],
               [1., 0.25093569],
               [1., 0.62368063],
               [1., 0.22218508],
               [1., 0.87093629],
               [1., 0.57206278],
               [1., 0.09099317],
               [1., 0.89240185],
               [1., 0.21845616],
               [1., 0.29347268],
               [1., 0.65169056],
               [1., 0.11715272],
               [1., 0.94927196],
               [1., 0.26998962],
               [1., 0.61982911],
               [1., 0.92096361],
               [1., 0.36986979],
               [1., 0.9305578],
               [1., 0.71236703],
               [1., 0.31308043],
               [1., 0.0968646],
               [1., 0.726384],
               [1., 0.50071205],
               [1., 0.32398836],
               [1., 0.77001287],
               [1., 0.81771262],
               [1., 0.52848984],
               [1., 0.86623918],
               [1., 0.25500704],
               [1., 0.54650686],
               [1., 0.33586952],
               [1., 0.45683848],
               [1., 0.32536201],
               [1., 0.65526249],
               [1., 0.41667614],
               [1., 0.2512252],
               [1., 0.14901902],
               [1., 0.05947808],
               [1., 0.23014044],
               [1., 0.09339056],
               [1., 0.55054291],
               [1., 0.28404749],
               [1., 0.31514742],
               [1., 0.60440087],
               [1., 0.15766473],
               [1., 0.63174691],
               [1., 0.40608228],
               [1., 0.57006629],
               [1., 0.53938759],
               [1., 0.78450796],
               [1., 0.84413932],
               [1., 0.03656797],
               [1., 0.90610783],
               [1., 0.78435973],
               [1., 0.18655042],
               [1., 0.610352],
               [1., 0.89612132],
               [1., 0.19053809],
               [1., 0.23681431],
               [1., 0.8828991],
               [1., 0.44193942],
               [1., 0.11454773],
               [1., 0.14716912],
               [1., 0.66118037]])

# [np.ones(100), np.random.rand(100), np.random.rand(100)]
H2 = np.array([[1., 0.10521948, 0.35927599],
               [1., 0.5398316, 0.21929855],
               [1., 0.97852169, 0.65573342],
               [1., 0.8324201, 0.51919011],
               [1., 0.99524297, 0.8625658],
               [1., 0.74137043, 0.35566975],
               [1., 0.77130113, 0.3024722],
               [1., 0.5042596, 0.68337775],
               [1., 0.52290331, 0.26131423],
               [1., 0.17693081, 0.36573229],
               [1., 0.67866957, 0.85749588],
               [1., 0.54025216, 0.7909514],
               [1., 0.97826568, 0.5893303],
               [1., 0.21279225, 0.72977643],
               [1., 0.77111568, 0.7191419],
               [1., 0.07113215, 0.06492206],
               [1., 0.50758486, 0.88722924],
               [1., 0.83743183, 0.790204],
               [1., 0.37434644, 0.0780652],
               [1., 0.60167862, 0.10941002],
               [1., 0.53980352, 0.39107312],
               [1., 0.20478355, 0.04299127],
               [1., 0.86651549, 0.29925723],
               [1., 0.97760234, 0.21122192],
               [1., 0.70988131, 0.18109618],
               [1., 0.00115266, 0.74752841],
               [1., 0.700718, 0.41936673],
               [1., 0.47155058, 0.27079443],
               [1., 0.95794133, 0.01692842],
               [1., 0.47115007, 0.93190715],
               [1., 0.69137784, 0.98149951],
               [1., 0.05334113, 0.8752315],
               [1., 0.26538605, 0.63309544],
               [1., 0.43615426, 0.03988823],
               [1., 0.58908664, 0.52999777],
               [1., 0.82853007, 0.43602184],
               [1., 0.45932654, 0.93333521],
               [1., 0.20802185, 0.64351782],
               [1., 0.68726465, 0.44290559],
               [1., 0.71875365, 0.62477263],
               [1., 0.03059299, 0.37685021],
               [1., 0.13695993, 0.9999803],
               [1., 0.68975949, 0.52104106],
               [1., 0.9389615, 0.10187587],
               [1., 0.92135372, 0.60209524],
               [1., 0.19434064, 0.38182183],
               [1., 0.68996819, 0.40138649],
               [1., 0.62765572, 0.02377619],
               [1., 0.59374729, 0.786097],
               [1., 0.2153557, 0.62052384],
               [1., 0.80332514, 0.96096576],
               [1., 0.24471115, 0.98096617],
               [1., 0.65190424, 0.26073113],
               [1., 0.85718719, 0.58069371],
               [1., 0.67380844, 0.93915817],
               [1., 0.90427549, 0.91982531],
               [1., 0.57098292, 0.25157481],
               [1., 0.46672042, 0.94076058],
               [1., 0.46002541, 0.80693844],
               [1., 0.75486451, 0.07751294],
               [1., 0.89425733, 0.75588687],
               [1., 0.12720157, 0.1391076],
               [1., 0.88904266, 0.03773833],
               [1., 0.14407169, 0.76585064],
               [1., 0.54354505, 0.51936926],
               [1., 0.3527304, 0.27549216],
               [1., 0.61370148, 0.25201815],
               [1., 0.39736844, 0.97335493],
               [1., 0.82779525, 0.56477333],
               [1., 0.6674417, 0.95022541],
               [1., 0.76968438, 0.35011643],
               [1., 0.25598441, 0.90349524],
               [1., 0.0334262, 0.00285389],
               [1., 0.84642894, 0.99756968],
               [1., 0.49823196, 0.28236571],
               [1., 0.50491213, 0.79108051],
               [1., 0.057909, 0.72875509],
               [1., 0.89215857, 0.75576954],
               [1., 0.86756251, 0.20579405],
               [1., 0.80905451, 0.70753031],
               [1., 0.73100967, 0.01807441],
               [1., 0.27674647, 0.78205623],
               [1., 0.03624884, 0.24957229],
               [1., 0.64759515, 0.73156287],
               [1., 0.13688036, 0.69747599],
               [1., 0.77714366, 0.29887156],
               [1., 0.50143917, 0.49325788],
               [1., 0.70702184, 0.53544101],
               [1., 0.37994049, 0.95749361],
               [1., 0.78272031, 0.24460296],
               [1., 0.32551035, 0.18652883],
               [1., 0.46446448, 0.40672246],
               [1., 0.58987784, 0.23863463],
               [1., 0.50670894, 0.14769436],
               [1., 0.25134429, 0.3968286],
               [1., 0.89777788, 0.32798564],
               [1., 0.8198818, 0.33782854],
               [1., 0.33083254, 0.59723932],
               [1., 0.08660821, 0.49089794],
               [1., 0.73677786, 0.33273433]])

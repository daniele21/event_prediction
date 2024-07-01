# -*- coding: utf-8 -*-
#%% SERIE_A

SERIE_A = 'serie_a'
SERIE_A_ID = 2019

SERIE_A_PATH = ['https://www.football-data.co.uk/mmz4281/1011/I1.csv',
               'https://www.football-data.co.uk/mmz4281/1112/I1.csv',
               'https://www.football-data.co.uk/mmz4281/1213/I1.csv',
               'https://www.football-data.co.uk/mmz4281/1314/I1.csv',
               'https://www.football-data.co.uk/mmz4281/1415/I1.csv',
               'https://www.football-data.co.uk/mmz4281/1516/I1.csv',
               'https://www.football-data.co.uk/mmz4281/1617/I1.csv',
               'https://www.football-data.co.uk/mmz4281/1718/I1.csv',
               'https://www.football-data.co.uk/mmz4281/1819/I1.csv',
               'https://www.football-data.co.uk/mmz4281/1920/I1.csv',
               'https://www.football-data.co.uk/mmz4281/2021/I1.csv']

PARMA = 'Parma'
FIORENTINA = 'Fiorentina'
UDINESE = 'Udinese'
CAGLIARI = 'Cagliari'
ROMA = 'Roma'
SAMPDORIA = 'Sampdoria'
SPAL = 'Spal'
TORINO = 'Torino'
VERONA = 'Verona'
INTER = 'Inter'
BOLOGNA = 'Bologna'
MILAN = 'Milan'
JUVENTUS = 'Juventus'
LAZIO = 'Lazio'
ATALANTA = 'Atalanta'
GENOA = 'Genoa'
LECCE = 'Lecce'
SASSUOLO = 'Sassuolo'
NAPOLI = 'Napoli'
BRESCIA = 'Brescia'
SPEZIA = 'Spezia'
BENEVENTO = 'Benevento'
CROTONE = 'Crotone'

SERIE_A_TEAMS = [PARMA, FIORENTINA, UDINESE, CAGLIARI, ROMA, SAMPDORIA,
                 TORINO, VERONA, INTER, BOLOGNA, MILAN, JUVENTUS,
                 LAZIO, ATALANTA, GENOA, SASSUOLO, NAPOLI,
                 SPEZIA, BENEVENTO, CROTONE]

SERIE_A_DECODER = {  'AC Milan': MILAN,
                     'ACF Fiorentina': FIORENTINA,
                     'AS Roma': ROMA,
                     'Atalanta BC': ATALANTA,
                     'Benevento Calcio': BENEVENTO,
                     'Bologna FC 1909': BOLOGNA,
                     'Cagliari Calcio': CAGLIARI,
                     'FC Crotone': CROTONE,
                     'FC Internazionale Milano': INTER,
                     'Genoa CFC': GENOA,
                     'Hellas Verona FC': VERONA,
                     'Juventus FC': JUVENTUS,
                     'Parma Calcio 1913': PARMA,
                     'SS Lazio': LAZIO,
                     'SSC Napoli': NAPOLI,
                     'Spezia Calcio': SPEZIA,
                     'Torino FC': TORINO,
                     'UC Sampdoria': SAMPDORIA,
                     'US Sassuolo Calcio': SASSUOLO,
                     'Udinese Calcio': UDINESE}

#%% PREMIER
PREMIER = 'premier_league'
PREMIER_ID = 2021

PREMIER_PATH = ['https://www.football-data.co.uk/mmz4281/1011/E0.csv',
               'https://www.football-data.co.uk/mmz4281/1112/E0.csv',
               'https://www.football-data.co.uk/mmz4281/1213/E0.csv',
               'https://www.football-data.co.uk/mmz4281/1314/E0.csv',
               'https://www.football-data.co.uk/mmz4281/1415/E0.csv',
               'https://www.football-data.co.uk/mmz4281/1516/E0.csv',
               'https://www.football-data.co.uk/mmz4281/1617/E0.csv',
               'https://www.football-data.co.uk/mmz4281/1718/E0.csv',
               'https://www.football-data.co.uk/mmz4281/1819/E0.csv',
               'https://www.football-data.co.uk/mmz4281/1920/E0.csv',
               'https://www.football-data.co.uk/mmz4281/2021/E0.csv']

LIVERPOOL = 'Liverpool'
WEST_HAM = 'West Ham'
BURNLEY = 'Burnley'
CRYSTAL_PALACE = 'Crystal Palace'
TOTTENHAM = 'Tottenham'
LEICESTER = 'Leicester'
NEWCASTLE = 'Newcastle'
MAN_UNITED = 'Man United'
ARSENAL = 'Arsenal'
ASTON_VILLA = 'Aston Villa'
BRIGHTON = 'Brighton'
EVERTON = 'Everton'
SOUTHAMPTON = 'Southampton'
MAN_CITY = 'Man City'
SHEFFIELD_UNITED = 'Sheffield United'
CHELSEA = 'Chelsea'
WOLVES = 'Wolves'
WEST_BROM = 'West Brom'
FULHAM = 'Fulham'
LEEDS = 'Leeds'

PREMIER_TEAMS = [LIVERPOOL, WEST_HAM, BURNLEY, CRYSTAL_PALACE, LEEDS, WEST_BROM,
                 TOTTENHAM, LEICESTER, NEWCASTLE, MAN_UNITED,
                 ARSENAL, ASTON_VILLA, BRIGHTON, EVERTON, SOUTHAMPTON,
                 MAN_CITY, SHEFFIELD_UNITED, CHELSEA, WOLVES, FULHAM]

PREMIER_DECODER = { 'Arsenal FC': ARSENAL,
                    'Aston Villa FC': ASTON_VILLA,
                    'Brighton & Hove Albion FC': BRIGHTON,
                    'Burnley FC': BURNLEY,
                    'Chelsea FC': CHELSEA,
                    'Crystal Palace FC':CRYSTAL_PALACE,
                    'Everton FC': EVERTON,
                    'Fulham FC': FULHAM,
                    'Leeds United FC': LEEDS,
                    'Leicester City FC': LEICESTER,
                    'Liverpool FC': LIVERPOOL,
                    'Manchester City FC': MAN_CITY,
                    'Manchester United FC': MAN_UNITED,
                    'Newcastle United FC': NEWCASTLE,
                    'Sheffield United FC': SHEFFIELD_UNITED,
                    'Southampton FC': SOUTHAMPTON,
                    'Tottenham Hotspur FC': TOTTENHAM,
                    'West Bromwich Albion FC': WEST_BROM,
                    'West Ham United FC': WEST_HAM,
                    'Wolverhampton Wanderers FC': WOLVES}

#%%
PREMIER_2 = 'premier_2'
PREMIER_2_ID = 2016

PREMIER_2_PATH = ['https://www.football-data.co.uk/mmz4281/1011/E1.csv',
                'https://www.football-data.co.uk/mmz4281/1112/E1.csv',
                'https://www.football-data.co.uk/mmz4281/1213/E1.csv',
                'https://www.football-data.co.uk/mmz4281/1314/E1.csv',
                'https://www.football-data.co.uk/mmz4281/1415/E1.csv',
                'https://www.football-data.co.uk/mmz4281/1516/E1.csv',
                'https://www.football-data.co.uk/mmz4281/1617/E1.csv',
                'https://www.football-data.co.uk/mmz4281/1819/E1.csv',
                'https://www.football-data.co.uk/mmz4281/1920/E1.csv',
                'https://www.football-data.co.uk/mmz4281/2021/E1.csv']

LUTON = 'Luton'
BOURNEMOUTH = 'Bournemouth'
BARNSLEY = 'Barnsley'
BLACKBURN = 'Blackburn'
BRENTFORD = 'Brentford'
COVENTRY = 'Coventry'
MILLWALL = 'Millwall'
READING = 'Reading'
STOKE = 'Stoke'
SWANSEA = 'Swansea'
WIGAN = 'Wigan'
NOTTINGAM_FOREST = 'Nott\'m Forest'
BRISTOL_CITY = 'Bristol City'
HUDDERSFIELD = 'Huddersfield'
LEEDS = 'Leeds'
BIRMINGHAM = 'Birmingham'
CARDIFF = 'Cardiff'
CHARLTON = 'Charlton'
DERBY = 'Derby'
FULHAM = 'Fulham'
HULL = 'Hull'
NORWICH = 'Norwich'
MIDDLESBROUGH = 'Middlesbrough'
PRESTON = 'Preston'
QPR = 'QPR'
SHEFFIELD_WEDS = 'Sheffield Weds'
WEST_BROM = 'West Brom'
WATFORD = 'Watford'
WYCOMBE = 'Wycombe'
ROTHERHAM = 'Rotherham'

PREMIER_2_TEAMS = [LUTON, BARNSLEY, BLACKBURN, BRENTFORD, MILLWALL,
                   READING, STOKE, SWANSEA, WIGAN, NOTTINGAM_FOREST,
                   BRISTOL_CITY, HUDDERSFIELD, LEEDS, BIRMINGHAM, CARDIFF,
                   CHARLTON, DERBY, FULHAM, HULL, MIDDLESBROUGH,
                   PRESTON, QPR, SHEFFIELD_WEDS, WEST_BROM]

PREMIER_2_DECODER = {'AFC Bournemouth': BOURNEMOUTH,
                     'Barnsley FC': BARNSLEY,
                     'Birmingham City FC': BIRMINGHAM,
                     'Blackburn Rovers FC': BLACKBURN,
                     'Brentford FC': BRENTFORD,
                     'Bristol City FC':BRISTOL_CITY,
                     'Cardiff City FC': CARDIFF,
                     'Coventry City FC': COVENTRY,
                     'Derby County FC': DERBY,
                     'Huddersfield Town AFC': HUDDERSFIELD,
                     'Luton Town FC': LUTON,
                     'Middlesbrough FC': MIDDLESBROUGH,
                     'Millwall FC': MILLWALL,
                     'Norwich City FC': NORWICH,
                     'Nottingham Forest FC': NOTTINGAM_FOREST,
                     'Preston North End FC': PRESTON,
                     'Queens Park Rangers FC': QPR,
                     'Reading FC': READING,
                     'Rotherham United FC': ROTHERHAM,
                     'Sheffield Wednesday FC': SHEFFIELD_WEDS,
                     'Stoke City FC': STOKE,
                     'Swansea City AFC': SWANSEA,
                     'Watford FC': WATFORD,
                     'Wycombe Wanderers FC': WYCOMBE}

#%%

JUPILIER = 'jupilier'
JUPILIER_ID = 2009

JUPILIER_PATH = ['https://www.football-data.co.uk/mmz4281/1011/B1.csv',
                 'https://www.football-data.co.uk/mmz4281/1112/B1.csv',
                 'https://www.football-data.co.uk/mmz4281/1213/B1.csv',
                 'https://www.football-data.co.uk/mmz4281/1314/B1.csv',
                 'https://www.football-data.co.uk/mmz4281/1415/B1.csv',
                 'https://www.football-data.co.uk/mmz4281/1516/B1.csv',
                 'https://www.football-data.co.uk/mmz4281/1617/B1.csv',
                 'https://www.football-data.co.uk/mmz4281/1718/B1.csv',
                 'https://www.football-data.co.uk/mmz4281/1819/B1.csv',
                 'https://www.football-data.co.uk/mmz4281/1920/B1.csv',
                 'https://www.football-data.co.uk/mmz4281/2021/B1.csv']

GENK = 'Genk'
CERCLE_BRUGGE = 'Cercle Brugge'
ST_TRUIDEN = 'St Truiden'
WAREGEM = 'Waregem'
WAASLAND_BEVEREN = 'Waasland-Beveren'
ANDERLECHT = 'Anderlecht'
CHARLEROI = 'Charleroi'
EUPEN = 'Eupen'
CLUB_BRUGGE = 'Club Brugge'
ST_LIEGI = 'St Liegi'
KORTRIJK = 'Kortrijk'
OOSTENDE = 'Oostende'
MECHELEN = 'Mechelen'
GENT = 'Gent'
MOUSCRON = 'Mouscron'
ANTWERP = 'Antwerp'
BEERSCHOT = 'Beerschot'
LEUVEN = 'Leuven'

JUPILIER_TEAMS = [BEERSCHOT, GENK, LEUVEN, CERCLE_BRUGGE, ST_TRUIDEN, WAREGEM, WAASLAND_BEVEREN,
                  ANDERLECHT, CHARLEROI, EUPEN, CLUB_BRUGGE, ST_LIEGI, 
                  KORTRIJK, OOSTENDE, MECHELEN, GENT, MOUSCRON, ANTWERP]


# LIGUE 1

LIGUE_1 = 'ligue_1'
LIGUE_1_ID = 2015

LIGUE_1_PATH = ['https://www.football-data.co.uk/mmz4281/1011/F1.csv',
                'https://www.football-data.co.uk/mmz4281/1112/F1.csv',
                'https://www.football-data.co.uk/mmz4281/1213/F1.csv',
                'https://www.football-data.co.uk/mmz4281/1314/F1.csv',
                'https://www.football-data.co.uk/mmz4281/1415/F1.csv',
                'https://www.football-data.co.uk/mmz4281/1516/F1.csv',
                'https://www.football-data.co.uk/mmz4281/1617/F1.csv',
                'https://www.football-data.co.uk/mmz4281/1718/F1.csv',
                'https://www.football-data.co.uk/mmz4281/1819/F1.csv',
                'https://www.football-data.co.uk/mmz4281/1920/F1.csv',
                'https://www.football-data.co.uk/mmz4281/2021/F1.csv']

MONACO = 'Monaco'
MARSIGLIA = 'Marsiglia'
ANGERS = 'Angers'
BREST = 'Brest'
DIJON = 'Dijon'
MONTPELLIER = 'Montpellier'
NIZZA = 'Nizza'
LENS = 'Lens'
LILLA = 'Lilla'
LORIENT = 'Lorient'
STRASBURGO = 'Strasburgo'
PARIS_SG = 'Paris SG'
LIONE = 'Lione'
NANTES = 'Nantes'
#AMIENS = 'Amiens'
BORDEAUX = 'Bordeaux'
METZ = 'Metz'
NIMES = 'Nimes'
#TOULOUSE = 'Toulouse'
ST_ETIENNE = 'St Etienne'
REIMS = 'Reims'
RENNES = 'Rennes'

LIGUE_1_TEAMS = [ANGERS, BORDEAUX, BREST, DIJON, LENS, LILLA, LIONE, LORIENT, MARSIGLIA, METZ,
                 MONACO, MONTPELLIER, NANTES, NIMES, NIZZA, PARIS_SG, REIMS, RENNES, ST_ETIENNE, STRASBURGO]

LIGUE_1_DECODER =  {'AS Monaco FC': MONACO,
                    'AS Saint-Étienne': ST_ETIENNE,
                    'Angers SCO': ANGERS,
                    "Dijon Football Côte d'Or": DIJON,
                    'FC Girondins de Bordeaux': BORDEAUX,
                    'FC Lorient': LORIENT,
                    'FC Metz': METZ,
                    'FC Nantes': NANTES,
                    'Lille OSC': LILLA,
                    'Montpellier HSC': MONTPELLIER,
                    'Nîmes Olympique': NIMES,
                    'OGC Nice': NIZZA,
                    'Olympique Lyonnais': LIONE,
                    'Olympique de Marseille': MARSIGLIA,
                    'Paris Saint-Germain FC': PARIS_SG,
                    'RC Strasbourg Alsace': STRASBURGO,
                    'Racing Club de Lens': LENS,
                    'Stade Brestois 29': BREST,
                    'Stade Rennais FC 1901': RENNES,
                    'Stade de Reims': REIMS}

# LIGUE 2
LIGUE_2 = 'ligue_2'
LIGUE_2_ID = 2142

LIGUE_2_PATH = ['https://www.football-data.co.uk/mmz4281/1011/F2.csv',
                 'https://www.football-data.co.uk/mmz4281/1112/F2.csv',
                 'https://www.football-data.co.uk/mmz4281/1213/F2.csv',
                 'https://www.football-data.co.uk/mmz4281/1314/F2.csv',
                 'https://www.football-data.co.uk/mmz4281/1415/F2.csv',
                 'https://www.football-data.co.uk/mmz4281/1516/F2.csv',
                 'https://www.football-data.co.uk/mmz4281/1617/F2.csv',
                 'https://www.football-data.co.uk/mmz4281/1718/F2.csv',
                 'https://www.football-data.co.uk/mmz4281/1819/F2.csv',
                 'https://www.football-data.co.uk/mmz4281/1920/F2.csv',
                 'https://www.football-data.co.uk/mmz4281/2021/F2.csv']

TOULOUSE = 'Toulouse'
AJACCIO = 'Ajaccio'
AMIENS = 'Amiens'
AUXERRE = 'Auxerre'
CHAMBLY = 'Chambly'
CLERMONT = 'Clermont'
GUINGAMP = 'Guingamp'
RODEZ = 'Rodez'
VALENCIENNES = 'Valenciennes'
TROYES = 'Troyes'
CAEN = 'Caen'
NANCY = 'Nancy'
CHATEAUROUX = 'Chateauroux'
DUNKERQUE = 'Dunkerque'
GRENOBLE = 'Grenoble'
LE_HAVRE = 'Le Havre'
NIORT = 'Niort'
PARIS_FC = 'Paris FC'
PAU_FC = 'Pau FC'
SOCHAUX = 'Sochaux'

LIGUE_2_TEAMS = [TOULOUSE, AJACCIO, AMIENS, AUXERRE, CHAMBLY,
                 CLERMONT, GUINGAMP, RODEZ, VALENCIENNES, TROYES,
                 CAEN, NANCY, CHATEAUROUX, DUNKERQUE, GRENOBLE,
                 LE_HAVRE, NIORT, PARIS_FC, PAU_FC, SOCHAUX]



# EREDIVISIE
EREDIVISIE = 'eredivisie'
EREDIVISIE_ID = 2003

EREDIVISIE_PATH = ['https://www.football-data.co.uk/mmz4281/1011/N1.csv',
                     'https://www.football-data.co.uk/mmz4281/1112/N1.csv',
                     'https://www.football-data.co.uk/mmz4281/1213/N1.csv',
                     'https://www.football-data.co.uk/mmz4281/1314/N1.csv',
                     'https://www.football-data.co.uk/mmz4281/1415/N1.csv',
                     'https://www.football-data.co.uk/mmz4281/1516/N1.csv',
                     'https://www.football-data.co.uk/mmz4281/1617/N1.csv',
                     'https://www.football-data.co.uk/mmz4281/1718/N1.csv',
                     'https://www.football-data.co.uk/mmz4281/1819/N1.csv',
                     'https://www.football-data.co.uk/mmz4281/1920/N1.csv',
                     'https://www.football-data.co.uk/mmz4281/2021/N1.csv']

ZWOLLE = 'Zwolle'
EMMEN = 'FC Emmen'
VITESSE = 'Vitesse'
TWENTE = 'Twente'
VENLO = 'VVV Venlo'
HERACLES = 'Heracles'
FEYENOORD = 'Feyenoord'
DEN_HAAG = 'Den Haag'
AZ = 'AZ Alkmaar'
SPARTA_ROTTERDAM = 'Sparta Rotterdam'
GRONINGEN = 'Groningen'
AJAX = 'Ajax'
WILLEM_II = 'Willem II'
SITTARD = 'For Sittard'
HEERENVEEN = 'Heerenveen'
WAALWIJK = 'Waalwijk'
UTRECHT = 'Utrecht'
PSV = 'PSV Eindhoven'

EREDIVISIE_TEAMS = [AJAX, AZ, DEN_HAAG, EMMEN, FEYENOORD, GRONINGEN, HEERENVEEN, HERACLES,
                    PSV, SITTARD, SPARTA_ROTTERDAM, TWENTE, UTRECHT, VENLO, VITESSE,
                    WAALWIJK, WILLEM_II, ZWOLLE]

EREDIVISIE_DECODER = {  'ADO Den Haag': DEN_HAAG,
                        'AFC Ajax':AJAX,
                        'AZ': AZ,
                        'FC Emmen': EMMEN,
                        'FC Groningen': GRONINGEN,
                        "FC Twente '65": TWENTE,
                        'FC Utrecht':UTRECHT,
                        'Feyenoord Rotterdam': FEYENOORD,
                        'Fortuna Sittard': SITTARD,
                        'Heracles Almelo': HERACLES,
                        'PEC Zwolle': ZWOLLE,
                        'PSV': PSV,
                        'RKC Waalwijk': WAALWIJK,
                        'SBV Vitesse':VITESSE,
                        'SC Heerenveen': HEERENVEEN,
                        'Sparta Rotterdam': SPARTA_ROTTERDAM,
                        'VVV Venlo': VENLO,
                        'Willem II Tilburg': WILLEM_II}

# LIGA
LIGA = 'liga'
LIGA_ID = 2014

LIGA_PATH = ['https://www.football-data.co.uk/mmz4281/1011/SP1.csv',
             'https://www.football-data.co.uk/mmz4281/1112/SP1.csv',
             'https://www.football-data.co.uk/mmz4281/1213/SP1.csv',
             'https://www.football-data.co.uk/mmz4281/1314/SP1.csv',
             'https://www.football-data.co.uk/mmz4281/1415/SP1.csv',
             'https://www.football-data.co.uk/mmz4281/1516/SP1.csv',
             'https://www.football-data.co.uk/mmz4281/1617/SP1.csv',
             'https://www.football-data.co.uk/mmz4281/1718/SP1.csv',
             'https://www.football-data.co.uk/mmz4281/1819/SP1.csv',
             'https://www.football-data.co.uk/mmz4281/1920/SP1.csv',
             'https://www.football-data.co.uk/mmz4281/2021/SP1.csv']

ATH_BILBAO = 'Ath Bilbao'
CADICE = 'Cadice'
CELTA = 'Celta Vigo'
ELCHE = 'Elche'
HUESCA = 'Huesca'
VALENCIA = 'Valencia'
MALLORCA = 'Mallorca'
LEGANES = 'Leganes'
VILLARREAL = 'Villarreal'
ALAVES = 'Alaves'
ESPANOL = 'Espanol'
BETIS = 'Betis'
ATH_MADRID = 'Ath Madrid'
GRANADA = 'Granada'
LEVANTE = 'Levante'
OSASUNA = 'Osasuna'
REAL_MADRID = 'Real Madrid'
GETAFE = 'Getafe'
BARCELLONA = 'Barcelona'
SIVIGLIA = 'Siviglia'
REAL_SOCIEDAD = 'Sociedad'
EIBAR = 'Eibar'
VALLADOLID = 'Valladolid'

LIGA_TEAMS = [ALAVES, ATH_BILBAO, ATH_MADRID, BARCELLONA, BETIS, CADICE, CELTA,
              EIBAR, ELCHE, GETAFE, GRANADA, HUESCA, LEVANTE, OSASUNA, REAL_MADRID,
              REAL_SOCIEDAD, SIVIGLIA, VALENCIA, VALLADOLID, VILLARREAL]

LIGA_DECODER = {'Athletic Club': ATH_BILBAO,
                'CA Osasuna':OSASUNA,
                'Club Atlético de Madrid': ATH_MADRID,
                'Cádiz CF': CADICE,
                'Deportivo Alavés': ALAVES,
                'Elche CF':ELCHE,
                'FC Barcelona': BARCELLONA,
                'Getafe CF':GETAFE,
                'Granada CF': GRANADA,
                'Levante UD': LEVANTE,
                'RC Celta de Vigo': CELTA,
                'Real Betis Balompié': BETIS,
                'Real Madrid CF': REAL_MADRID,
                'Real Sociedad de Fútbol': REAL_SOCIEDAD,
                'Real Valladolid CF': VALLADOLID,
                'SD Eibar': EIBAR,
                'SD Huesca': HUESCA,
                'Sevilla FC': SIVIGLIA,
                'Valencia CF': VALENCIA,
                'Villarreal CF': VILLARREAL}

# LIGA_2
LIGA_2 = 'liga_2'
LIGA_2_ID = 2077

LIGA_2_PATH = ['https://www.football-data.co.uk/mmz4281/1011/SP2.csv',
                 'https://www.football-data.co.uk/mmz4281/1112/SP2.csv',
                 'https://www.football-data.co.uk/mmz4281/1213/SP2.csv',
                 'https://www.football-data.co.uk/mmz4281/1314/SP2.csv',
                 'https://www.football-data.co.uk/mmz4281/1415/SP2.csv',
                 'https://www.football-data.co.uk/mmz4281/1516/SP2.csv',
                 'https://www.football-data.co.uk/mmz4281/1617/SP2.csv',
                 'https://www.football-data.co.uk/mmz4281/1718/SP2.csv',
                 'https://www.football-data.co.uk/mmz4281/1819/SP2.csv',
                 'https://www.football-data.co.uk/mmz4281/1920/SP2.csv',
                 'https://www.football-data.co.uk/mmz4281/2021/SP2.csv']

LUGO = 'Lugo'
SANTANDER = 'Santander'
ALMERIA = 'Almeria'
CARTAGENA = 'Cartagena'
CASTELLON = 'Castellon'
ESPANYOL = 'Espanyol'
ELCHE = 'Elche'
LOGRONES = 'Logrones'
MAIORCA = 'Maiorca'
VALLECANO = 'Vallecano'
SABADELL = 'Sabadell'
SARAGOZZA = 'Zaragoza'
# DEP_LA_CORUNA = 'La Coruna'
NUMANCIA = 'Numancia'
GIRONA = 'Girona'
CADIZ = 'Cadiz'
LAS_PALMAS = 'Las Palmas'
ALBACETE = 'Albacete'
OVIEDO = 'Oviedo'
MIRANDES = 'Mirandes'
ALCORON = 'Alcorcon'
MALAGA = 'Malaga'
PONFERRADINA = 'Ponferradina'
GIJON = 'Sp Gijon'
HUESCA = 'Huesca'
TENERIFE = 'Tenerife'
EXTREMADURA = 'Extremadura UD'
FUENLABRADA = 'Fuenlabrada'
# RACING_SANTANDER = 'Racing Santander'

LIGA_2_TEAMS = [ALBACETE, ALCORON, ALMERIA, CARTAGENA, CASTELLON, ESPANYOL, FUENLABRADA,
                GIJON, GIRONA, LAS_PALMAS, LEGANES, LOGRONES, LUGO, MAIORCA, MALAGA, MIRANDES,
                PONFERRADINA, OVIEDO, SABADELL, SARAGOZZA, TENERIFE, VALLECANO]

# BUNDESLIGA
BUNDESLIGA = 'bundesliga'
BUNDESLIGA_ID = 2002

BUNDESLIGA_PATH = ['https://www.football-data.co.uk/mmz4281/1011/D1.csv',
                     'https://www.football-data.co.uk/mmz4281/1112/D1.csv',
                     'https://www.football-data.co.uk/mmz4281/1213/D1.csv',
                     'https://www.football-data.co.uk/mmz4281/1314/D1.csv',
                     'https://www.football-data.co.uk/mmz4281/1415/D1.csv',
                     'https://www.football-data.co.uk/mmz4281/1516/D1.csv',
                     'https://www.football-data.co.uk/mmz4281/1617/D1.csv',
                     'https://www.football-data.co.uk/mmz4281/1718/D1.csv',
                     'https://www.football-data.co.uk/mmz4281/1819/D1.csv',
                     'https://www.football-data.co.uk/mmz4281/1920/D1.csv',
                     'https://www.football-data.co.uk/mmz4281/2021/D1.csv']

AUGSBURG = 'Augsburg'
BAYERN_MUNICH = 'Bayern Munich'
BIELEFELD = 'Bielefeld'
DORTMUND = 'Dortmund'
EIN_FRANKFURT = 'Ein Frankfurt'
FC_KOLN = 'FC Koln'
FREIBURG = 'Freiburg'
HERTHA = 'Hertha'
HOFFENHEIM = 'Hoffenheim'
INGOLSTADT = 'Ingolstadt'
KAISERSLAUTERN = 'Kaiserslautern'
LEVERKUSEN = 'Leverkusen'
MGLADBACH = 'M\'gladbach'
MAINZ = 'Mainz'
RB_LEIPZIG = 'RB Leipzig'
SCHALKE_04 = 'Schalke 04'
STUTTGART = 'Stuttgart'
UNION_BERLIN = 'Union Berlin'
WERDER_BREMEN = 'Werder Bremen'
WOLFSBURG = 'Wolfsburg'

BUNDESLIGA_TEAMS = [BAYERN_MUNICH, EIN_FRANKFURT, FC_KOLN, STUTTGART,
                    UNION_BERLIN, WERDER_BREMEN, DORTMUND, RB_LEIPZIG,
                    WOLFSBURG, HERTHA, AUGSBURG, BIELEFELD, LEVERKUSEN,
                    MAINZ, MGLADBACH, SCHALKE_04, HOFFENHEIM, FREIBURG]

BUNDESLIGA_DECODER = {  '1. FC Köln': FC_KOLN,
                        '1. FC Union Berlin': UNION_BERLIN,
                        '1. FSV Mainz 05': MAINZ,
                        'Bayer 04 Leverkusen': LEVERKUSEN,
                        'Borussia Dortmund': DORTMUND,
                        'Borussia Mönchengladbach': MGLADBACH,
                        'DSC Arminia Bielefeld': BIELEFELD,
                        'Eintracht Frankfurt': EIN_FRANKFURT,
                        'FC Augsburg': AUGSBURG,
                        'FC Bayern München': BAYERN_MUNICH,
                        'FC Schalke 04': SCHALKE_04,
                        'Hertha BSC': HERTHA,
                        'RB Leipzig': RB_LEIPZIG,
                        'SC Freiburg': FREIBURG,
                        'SV Werder Bremen': WERDER_BREMEN,
                        'TSG 1899 Hoffenheim': HOFFENHEIM,
                        'VfB Stuttgart': STUTTGART,
                        'VfL Wolfsburg': WOLFSBURG}

# BUNDESLIGA_2

BUNDESLIGA_2 = 'bundesliga_2'
BUNDESLIGA_2_ID = 2004

BUNDESLIGA_2_PATH = ['https://www.football-data.co.uk/mmz4281/1011/D2.csv',
                     'https://www.football-data.co.uk/mmz4281/1112/D2.csv',
                     'https://www.football-data.co.uk/mmz4281/1213/D2.csv',
                     'https://www.football-data.co.uk/mmz4281/1314/D2.csv',
                     'https://www.football-data.co.uk/mmz4281/1415/D2.csv',
                     'https://www.football-data.co.uk/mmz4281/1516/D2.csv',
                     'https://www.football-data.co.uk/mmz4281/1617/D2.csv',
                     'https://www.football-data.co.uk/mmz4281/1718/D2.csv',
                     'https://www.football-data.co.uk/mmz4281/1819/D2.csv',
                     'https://www.football-data.co.uk/mmz4281/1920/D2.csv',
                     'https://www.football-data.co.uk/mmz4281/2021/D2.csv']

HAMBURG = 'Hamburg'
REGENSBURG = 'Regensburg'
HANNOVER = 'Hannover'
SANDHAUSEN = 'Sandhausen'
WURZBURGER_KICKERS = 'Wurzburger Kickers'
GREUTHER_FURTH = 'Greuther Furth'
HEIDENHEIM = 'Heidenheim'
HOLSTEIN_KIEL = 'Holstein Kiel'
BOCHUM = 'Bochum'
ERZGEBIRGE_AUE = 'Erzgebirge Aue'
OSNABRUCK = 'Osnabruck'
BRAUNSCHWEIG = 'Braunschweig'
DARMSTADT = 'Darmstadt'
FORTUNA_DUSSELDORF = 'Fortuna Dusseldorf'
KARLSRUHE = 'Karlsruhe'
NURNBERG = 'Nurnberg'
ST_PAULI = 'St Pauli'
PADERBORN = 'Paderborn'

BUNDESLIGA_2_TEAMS = [HAMBURG, REGENSBURG, HANNOVER, SANDHAUSEN, WURZBURGER_KICKERS, GREUTHER_FURTH,
                      HEIDENHEIM, HOLSTEIN_KIEL, BOCHUM, ERZGEBIRGE_AUE, OSNABRUCK, BRAUNSCHWEIG,
                      DARMSTADT, FORTUNA_DUSSELDORF, KARLSRUHE, NURNBERG, ST_PAULI, PADERBORN]

#---------------------------------------------------------------------------------------------------------------------------------------------
#%%
LEAGUE_NAMES = [SERIE_A,
                PREMIER,
                PREMIER_2,
                LIGUE_1,
                LIGUE_2,
                JUPILIER,
                EREDIVISIE,
                LIGA,
                LIGA_2,
                BUNDESLIGA,
                BUNDESLIGA_2,
                ]

LEAGUE_PATHS = {SERIE_A: SERIE_A_PATH,
                PREMIER: PREMIER_PATH,
                PREMIER_2: PREMIER_2_PATH,
                EREDIVISIE: EREDIVISIE_PATH,
                LIGUE_1: LIGUE_1_PATH,
                LIGUE_2: LIGUE_2_PATH,
                LIGA: LIGA_PATH,
                LIGA_2: LIGA_2_PATH,
                JUPILIER: JUPILIER_PATH,
		        BUNDESLIGA: BUNDESLIGA_PATH,
                BUNDESLIGA_2: BUNDESLIGA_2_PATH}

TEAMS_LEAGUE = {SERIE_A : SERIE_A_TEAMS,
                PREMIER : PREMIER_TEAMS,
                PREMIER_2: PREMIER_2_TEAMS,
                JUPILIER: JUPILIER_TEAMS,
                LIGUE_1: LIGUE_1_TEAMS,
                LIGUE_2: LIGUE_2_TEAMS,
                EREDIVISIE: EREDIVISIE_TEAMS,
                LIGA: LIGA_TEAMS,
                LIGA_2: LIGA_2_TEAMS,
                BUNDESLIGA: BUNDESLIGA_TEAMS,
                BUNDESLIGA_2: BUNDESLIGA_2_TEAMS
                }

N_TEAMS = {SERIE_A : 20,
           PREMIER : 20,
           PREMIER_2 : 24,
           JUPILIER: 18,
           LIGUE_1: 20,
           LIGUE_2: 20,
           EREDIVISIE: 18,
           LIGA: 20,
           LIGA_2: 22,
           BUNDESLIGA: 18,
           BUNDESLIGA_2: 18}

LEAGUE_ENCODER = {'league2id': {PREMIER: PREMIER_ID,
                                PREMIER_2: PREMIER_2_ID,
                                SERIE_A: SERIE_A_ID,
                                BUNDESLIGA: BUNDESLIGA_ID,
                                BUNDESLIGA_2: BUNDESLIGA_2_ID,
                                JUPILIER: JUPILIER_ID,
                                EREDIVISIE: EREDIVISIE_ID,
                                LIGUE_1: LIGUE_1_ID,
                                LIGUE_2: LIGUE_2_ID,
                                LIGA: LIGA_ID,
                                LIGA_2: LIGA_2_ID},
           }

LEAGUE_DECODER = {'id2league': {PREMIER_ID: PREMIER,
                                PREMIER_2_ID: PREMIER_2,
                                SERIE_A_ID: SERIE_A ,
                                BUNDESLIGA_ID: BUNDESLIGA,
                                BUNDESLIGA_2_ID: BUNDESLIGA_2,
                                JUPILIER_ID: JUPILIER,
                                EREDIVISIE_ID: EREDIVISIE,
                                LIGUE_1_ID: LIGUE_1,
                                LIGUE_2_ID: LIGUE_2,
                                LIGA_ID: LIGA,
                                LIGA_2_ID: LIGA_2
                                },
                   'league2league': {PREMIER: PREMIER_DECODER,
                                     SERIE_A: SERIE_A_DECODER,
                                     PREMIER_2: PREMIER_2_DECODER,
                                     LIGUE_1: LIGUE_1_DECODER,
                                     EREDIVISIE: EREDIVISIE_DECODER,
                                     BUNDESLIGA: BUNDESLIGA_DECODER,
                                     LIGA: LIGA_DECODER}
                  }
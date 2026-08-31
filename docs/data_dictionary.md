# Formula 1 Race Prediction System: Comprehensive Data Dictionary

This document outlines the raw historical F1 dataset schemas, entity relationships, primary/foreign keys, and engineered machine learning features.

## 1. Raw Dataset Tables Overview

| Table Name | Rows | Columns | Primary Key | Description |
| :--- | :--- | :--- | :--- | :--- |
| `circuits.csv` | 78 | 9 | `circuitId` | Grand Prix circuit metadata (coordinates, altitude, country, track ref) |
| `constructor_results.csv` | 12,997 | 5 | `constructorResultsId` | Constructor race point tallies and DNF statuses per event |
| `constructor_standings.csv` | 13,763 | 7 | `constructorStandingsId` | Constructor Championship points and standings after each round |
| `constructors.csv` | 214 | 5 | `constructorId` | F1 teams / constructors (name, nationality, constructorRef) |
| `driver_standings.csv` | 35,626 | 7 | `driverStandingsId` | Driver Championship points, rank, and win tallies after each round |
| `drivers.csv` | 865 | 9 | `driverId` | Driver bio data (names, driverRef, DOB, nationality, code, number) |
| `lap_times.csv` | 879,870 | 6 | `raceId` | Lap-by-lap timing records (lap, position, milliseconds) across races |
| `pit_stops.csv` | 22,613 | 7 | `raceId` | Pit stop event records (stop number, lap, duration, milliseconds) |
| `qualifying.csv` | 11,234 | 9 | `qualifyId` | Qualifying classification records (Q1, Q2, Q3 lap times, grid position) |
| `races.csv` | 1,172 | 18 | `raceId` | Grand Prix event schedule, circuit references, dates, and session times |
| `results.csv` | 27,502 | 18 | `resultId` | Official Grand Prix finishing classification, points, grid, laps, status |
| `seasons.csv` | 77 | 2 | `year` | F1 championship season list (1950 to present) |
| `sprint_results.csv` | 590 | 17 | `resultId` | Sprint race classification and points records |
| `status.csv` | 141 | 2 | `statusId` | Race finishing status reference (Finished, Engine, Collision, etc.) |

## 2. Table Schemas & Foreign Key Relationships

### `circuits.csv`
**Shape:** `78` rows × `9` columns

| Column | Data Type | Non-Null % | Key Type | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| `circuitId` | `int64` | 100.0% | **Primary Key** | 1, 2, 3 |
| `circuitRef` | `object` | 100.0% | Normal | albert_park, sepang, bahrain |
| `name` | `object` | 100.0% | Normal | Albert Park Grand Prix Circuit, Sepan... |
| `location` | `object` | 100.0% | Normal | Melbourne, Kuala Lumpur, Sakhir |
| `country` | `object` | 100.0% | Normal | Australia, Malaysia, Bahrain |
| `lat` | `float64` | 100.0% | Normal | -37.8497, 2.76083, 26.0325 |
| `lng` | `float64` | 100.0% | Normal | 144.968, 101.738, 50.5106 |
| `alt` | `int64` | 100.0% | Normal | 10, 18, 7 |
| `url` | `object` | 100.0% | Normal | http://en.wikipedia.org/wiki/Melbourn... |


### `constructor_results.csv`
**Shape:** `12,997` rows × `5` columns

| Column | Data Type | Non-Null % | Key Type | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| `constructorResultsId` | `int64` | 100.0% | *Foreign Key* | 1, 2, 3 |
| `raceId` | `int64` | 100.0% | *Foreign Key* | 18, 19, 20 |
| `constructorId` | `int64` | 100.0% | *Foreign Key* | 1, 2, 3 |
| `points` | `float64` | 100.0% | Normal | 14.0, 8.0, 9.0 |
| `status` | `object` | 0.1% | Normal | D |


### `constructor_standings.csv`
**Shape:** `13,763` rows × `7` columns

| Column | Data Type | Non-Null % | Key Type | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| `constructorStandingsId` | `int64` | 100.0% | *Foreign Key* | 1, 2, 3 |
| `raceId` | `int64` | 100.0% | *Foreign Key* | 18, 19, 20 |
| `constructorId` | `int64` | 100.0% | *Foreign Key* | 1, 2, 3 |
| `points` | `float64` | 100.0% | Normal | 14.0, 8.0, 9.0 |
| `position` | `float64` | 100.0% | Normal | 1.0, 3.0, 2.0 |
| `positionText` | `object` | 100.0% | Normal | 1, 3, 2 |
| `wins` | `int64` | 100.0% | Normal | 1, 0, 2 |


### `constructors.csv`
**Shape:** `214` rows × `5` columns

| Column | Data Type | Non-Null % | Key Type | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| `constructorId` | `int64` | 100.0% | **Primary Key** | 1, 2, 3 |
| `constructorRef` | `object` | 100.0% | Normal | mclaren, bmw_sauber, williams |
| `name` | `object` | 100.0% | Normal | McLaren, BMW Sauber, Williams |
| `nationality` | `object` | 100.0% | Normal | British, German, French |
| `url` | `object` | 100.0% | Normal | http://en.wikipedia.org/wiki/McLaren,... |


### `driver_standings.csv`
**Shape:** `35,626` rows × `7` columns

| Column | Data Type | Non-Null % | Key Type | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| `driverStandingsId` | `int64` | 100.0% | *Foreign Key* | 1, 2, 3 |
| `raceId` | `int64` | 100.0% | *Foreign Key* | 18, 19, 20 |
| `driverId` | `int64` | 100.0% | *Foreign Key* | 1, 2, 3 |
| `points` | `float64` | 100.0% | Normal | 10.0, 8.0, 6.0 |
| `position` | `float64` | 100.0% | Normal | 1.0, 2.0, 3.0 |
| `positionText` | `object` | 100.0% | Normal | 1, 2, 3 |
| `wins` | `int64` | 100.0% | Normal | 1, 0, 2 |


### `drivers.csv`
**Shape:** `865` rows × `9` columns

| Column | Data Type | Non-Null % | Key Type | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| `driverId` | `int64` | 100.0% | **Primary Key** | 1, 2, 3 |
| `driverRef` | `object` | 100.0% | Normal | hamilton, heidfeld, rosberg |
| `number` | `float64` | 7.3% | Normal | 44.0, 6.0, 14.0 |
| `code` | `object` | 12.5% | Normal | HAM, HEI, ROS |
| `forename` | `object` | 100.0% | Normal | Lewis, Nick, Nico |
| `surname` | `object` | 100.0% | Normal | Hamilton, Heidfeld, Rosberg |
| `dob` | `object` | 100.0% | Normal | 1985-01-07, 1977-05-10, 1985-06-27 |
| `nationality` | `object` | 100.0% | Normal | British, German, Spanish |
| `url` | `object` | 100.0% | Normal | http://en.wikipedia.org/wiki/Lewis_Ha... |


### `lap_times.csv`
**Shape:** `879,870` rows × `6` columns

| Column | Data Type | Non-Null % | Key Type | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| `raceId` | `int64` | 100.0% | *Foreign Key* | 479, 471, 468 |
| `driverId` | `int64` | 100.0% | *Foreign Key* | 137, 119, 105 |
| `lap` | `int64` | 100.0% | Normal | 1, 2, 3 |
| `position` | `int64` | 100.0% | Normal | 1, 2, 4 |
| `time` | `object` | 100.0% | Normal | 1:42.085, 1:36.287, 1:34.627 |
| `milliseconds` | `int64` | 100.0% | Normal | 102085, 96287, 94627 |


### `pit_stops.csv`
**Shape:** `22,613` rows × `7` columns

| Column | Data Type | Non-Null % | Key Type | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| `raceId` | `int64` | 100.0% | *Foreign Key* | 258, 259, 261 |
| `driverId` | `int64` | 100.0% | *Foreign Key* | 100, 79, 57 |
| `stop` | `int64` | 100.0% | Normal | 1, 2, 3 |
| `lap` | `int64` | 100.0% | Normal | 1, 17, 18 |
| `time` | `object` | 100.0% | Normal | 14:01:34, 14:20:46, 14:22:35 |
| `duration` | `object` | 100.0% | Normal | 49.111, 28.482, 43.745 |
| `milliseconds` | `float64` | 100.0% | Normal | 49111.0, 28482.0, 43745.0 |


### `qualifying.csv`
**Shape:** `11,234` rows × `9` columns

| Column | Data Type | Non-Null % | Key Type | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| `qualifyId` | `int64` | 100.0% | *Foreign Key* | 1, 2, 3 |
| `raceId` | `int64` | 100.0% | *Foreign Key* | 18, 19, 20 |
| `driverId` | `int64` | 100.0% | *Foreign Key* | 1, 9, 5 |
| `constructorId` | `int64` | 100.0% | *Foreign Key* | 1, 2, 6 |
| `number` | `int64` | 100.0% | Normal | 22, 4, 23 |
| `position` | `int64` | 100.0% | Normal | 1, 2, 3 |
| `q1` | `object` | 98.5% | Normal | 1:26.572, 1:26.103, 1:25.664 |
| `q2` | `object` | 56.9% | Normal | 1:25.187, 1:25.315, 1:25.452 |
| `q3` | `object` | 35.4% | Normal | 1:26.714, 1:26.869, 1:27.079 |


### `races.csv`
**Shape:** `1,172` rows × `18` columns

| Column | Data Type | Non-Null % | Key Type | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| `raceId` | `int64` | 100.0% | **Primary Key** | 833, 834, 835 |
| `year` | `int64` | 100.0% | Normal | 1950, 1951, 1952 |
| `round` | `int64` | 100.0% | Normal | 1, 2, 3 |
| `circuitId` | `int64` | 100.0% | *Foreign Key* | 9, 6, 19 |
| `name` | `object` | 100.0% | Normal | British Grand Prix, Monaco Grand Prix... |
| `date` | `datetime64[ns]` | 100.0% | Normal | 1950-05-13 00:00:00, 1950-05-21 00:00... |
| `time` | `object` | 37.6% | Normal | 14:00:00, 15:00:00, 14:30:00 |
| `url` | `object` | 100.0% | Normal | http://en.wikipedia.org/wiki/1950_Bri... |
| `fp1_date` | `object` | 11.7% | Normal | 2021-03-26, 2021-04-16, 2021-04-30 |
| `fp1_time` | `object` | 9.8% | Normal | 12:00:00, 14:00:00, 03:00:00 |
| `fp2_date` | `object` | 10.2% | Normal | 2021-03-26, 2021-04-16, 2021-04-30 |
| `fp2_time` | `object` | 8.3% | Normal | 15:00:00, 17:00:00, 06:00:00 |
| `fp3_date` | `object` | 9.1% | Normal | 2021-03-27, 2021-04-17, 2021-05-01 |
| `fp3_time` | `object` | 7.5% | Normal | 12:00:00, 14:00:00, 03:00:00 |
| `quali_date` | `object` | 11.7% | Normal | 2021-03-27, 2021-04-17, 2021-05-01 |
| `quali_time` | `object` | 9.8% | Normal | 15:00:00, 17:00:00, 06:00:00 |
| `sprint_date` | `object` | 2.6% | Normal | 2021-07-17, 2021-09-11, 2021-11-13 |
| `sprint_time` | `object` | 2.3% | Normal | 14:30:00, 19:30:00, 13:30:00 |


### `results.csv`
**Shape:** `27,502` rows × `18` columns

| Column | Data Type | Non-Null % | Key Type | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| `resultId` | `int64` | 100.0% | **Primary Key** | 1, 2, 3 |
| `raceId` | `int64` | 100.0% | *Foreign Key* | 18, 19, 20 |
| `driverId` | `int64` | 100.0% | *Foreign Key* | 1, 2, 3 |
| `constructorId` | `int64` | 100.0% | *Foreign Key* | 1, 2, 3 |
| `number` | `float64` | 100.0% | Normal | 22.0, 3.0, 7.0 |
| `grid` | `int64` | 100.0% | *Foreign Key* | 1, 5, 7 |
| `position` | `float64` | 60.2% | Normal | 1.0, 2.0, 3.0 |
| `positionText` | `object` | 100.0% | Normal | 1, 2, 3 |
| `positionOrder` | `int64` | 100.0% | Normal | 1, 2, 3 |
| `points` | `float64` | 100.0% | Normal | 10.0, 8.0, 6.0 |
| `laps` | `int64` | 100.0% | Normal | 58, 57, 55 |
| `time` | `object` | 29.6% | Normal | 1:34:50.616, +5.478, +8.163 |
| `milliseconds` | `float64` | 29.6% | Normal | 5690616.0, 5696094.0, 5698779.0 |
| `fastestLap` | `float64` | 32.6% | Normal | 39.0, 41.0, 58.0 |
| `rank` | `float64` | 33.5% | Normal | 2.0, 3.0, 5.0 |
| `fastestLapTime` | `object` | 32.6% | Normal | 1:27.452, 1:27.739, 1:28.090 |
| `fastestLapSpeed` | `float64` | 30.0% | Normal | 218.3, 217.586, 216.719 |
| `statusId` | `int64` | 100.0% | *Foreign Key* | 1, 11, 5 |


### `seasons.csv`
**Shape:** `77` rows × `2` columns

| Column | Data Type | Non-Null % | Key Type | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| `year` | `int64` | 100.0% | Normal | 2009, 2008, 2007 |
| `url` | `object` | 100.0% | Normal | http://en.wikipedia.org/wiki/2009_For... |


### `sprint_results.csv`
**Shape:** `590` rows × `17` columns

| Column | Data Type | Non-Null % | Key Type | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| `resultId` | `int64` | 100.0% | *Foreign Key* | 1, 2, 3 |
| `raceId` | `int64` | 100.0% | *Foreign Key* | 1061, 1065, 1071 |
| `driverId` | `int64` | 100.0% | *Foreign Key* | 830, 1, 822 |
| `constructorId` | `int64` | 100.0% | *Foreign Key* | 9, 131, 6 |
| `number` | `int64` | 100.0% | Normal | 33, 44, 77 |
| `grid` | `int64` | 100.0% | *Foreign Key* | 2, 1, 3 |
| `position` | `float64` | 97.5% | Normal | 1.0, 2.0, 3.0 |
| `positionText` | `object` | 100.0% | Normal | 1, 2, 3 |
| `positionOrder` | `int64` | 100.0% | Normal | 1, 2, 3 |
| `points` | `int64` | 100.0% | Normal | 3, 2, 1 |
| `laps` | `int64` | 100.0% | Normal | 17, 16, 18 |
| `time` | `object` | 93.4% | Normal | 25:38.426, +1.430, +7.502 |
| `milliseconds` | `float64` | 93.4% | Normal | 1538426.0, 1539856.0, 1545928.0 |
| `fastestLap` | `float64` | 97.3% | Normal | 14.0, 17.0, 16.0 |
| `fastestLapTime` | `object` | 97.3% | Normal | 1:30.013, 1:29.937, 1:29.958 |
| `statusId` | `float64` | 99.7% | *Foreign Key* | 1.0, 76.0, 3.0 |
| `rank` | `float64` | 37.8% | Normal | 1.0, 4.0, 2.0 |


### `status.csv`
**Shape:** `141` rows × `2` columns

| Column | Data Type | Non-Null % | Key Type | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| `statusId` | `int64` | 100.0% | *Foreign Key* | 1, 2, 3 |
| `status` | `object` | 100.0% | Normal | Finished, Disqualified, Accident |


## 3. Engineered Machine Learning Feature Dictionary

All engineered features are strictly temporal: for any race at date $T$, only data strictly prior to $T$ ($t < T$) is used.

| Feature Name | Group | Type | Description | Leakage Prevention Rule |
| :--- | :--- | :--- | :--- | :--- |
| `driver_form_ewm_finish` | Driver Form | `float` | Exponentially weighted moving average of finishing position | Calculated strictly on prior races (shift 1) |
| `driver_rolling_finish_last3` | Driver Form | `float` | Mean finishing position across previous 3 races | Prior races only |
| `driver_rolling_finish_last5` | Driver Form | `float` | Mean finishing position across previous 5 races | Prior races only |
| `driver_season_avg_finish` | Driver Form | `float` | Season-to-date average finishing position | Rounds $1 \dots (k-1)$ of current season |
| `driver_career_avg_finish` | Driver Form | `float` | Career-to-date average finishing position | All historical career starts prior to current round |
| `driver_recent_points_sum5` | Driver Form | `float` | Total championship points scored in last 5 races | Prior 5 races |
| `driver_championship_stand_pos` | Driver Form | `float` | Championship standing position entering the weekend | Previous race standings table |
| `driver_championship_points` | Driver Form | `float` | Championship points accumulated entering the weekend | Previous race standings table |
| `driver_career_podium_rate` | Driver Track Record | `float` | Proportion of career starts resulting in top-3 finish | Strictly past races |
| `driver_career_win_rate` | Driver Track Record | `float` | Proportion of career starts resulting in P1 victory | Strictly past races |
| `driver_career_top10_rate` | Driver Track Record | `float` | Proportion of career starts resulting in top-10 finish | Strictly past races |
| `driver_career_dnf_rate` | Driver Reliability | `float` | Historical rate of race retirements (status != Finished/Lapped) | Past starts |
| `driver_recent_dnf_rate_5` | Driver Reliability | `float` | Rate of retirements in the last 5 races | Prior 5 starts |
| `driver_monza_starts` | Monza Circuit | `int` | Total career race starts at Autodromo Nazionale di Monza | Prior Monza GPs |
| `driver_monza_avg_finish` | Monza Circuit | `float` | Average finishing position specifically at Monza | Prior Monza GPs (imputed if 0 starts) |
| `driver_monza_podiums` | Monza Circuit | `int` | Career podium finishes at Monza | Prior Monza GPs |
| `driver_monza_dnf_rate` | Monza Circuit | `float` | Historical DNF rate specifically at Monza | Prior Monza GPs |
| `driver_monza_recency_weighted_finish` | Monza Circuit | `float` | Half-life weighted Monza performance (recent years heavier) | Time-decay on prior Monza GPs |
| `constructor_rolling_points_last5` | Constructor Pace | `float` | Constructor championship points in last 5 rounds | Prior constructor race points |
| `constructor_recent_avg_finish` | Constructor Pace | `float` | Constructor average finishing position across both cars (last 5) | Prior races |
| `constructor_season_points` | Constructor Pace | `float` | Current constructor championship points entering race | Prior standings |
| `constructor_season_rank` | Constructor Pace | `float` | Constructor championship rank entering race | Prior standings |
| `constructor_career_podium_rate` | Constructor Pace | `float` | Historical constructor podium rate | Prior races |
| `constructor_monza_avg_finish` | Constructor Circuit | `float` | Constructor historical average finish at Monza | Prior Monza races |
| `constructor_monza_podium_rate` | Constructor Circuit | `float` | Constructor historical podium rate at Monza | Prior Monza races |
| `constructor_reliability_dnf_rate` | Constructor Reliability | `float` | Constructor mechanical/overall DNF rate over last 10 races | Prior 10 constructor starts |
| `driver_quali_rolling_avg_last5` | Qualifying Form | `float` | Average qualifying position over prior 5 sessions | Prior qualifying sessions |
| `driver_quali_vs_teammate_diff` | Qualifying Form | `float` | Driver average qualifying position delta against direct teammate | Prior qualifying sessions |
| `grid_position` | Grid (Post-Quali) | `int` | Starting grid position on race day (2026 Monza confirmed/simulated) | Set to expected grid pre-quali, actual post-quali |
| `driver_lap_pace_rel_median` | Lap Pace | `float` | Historical race pace relative to field median from `lap_times.csv` | Prior aggregated race laps |
| `driver_lap_pace_std` | Lap Pace | `float` | Historical lap-time standard deviation (consistency) | Prior aggregated race laps |
| `constructor_pit_duration_mean` | Pit Stop | `float` | Constructor mean pit stop stationary duration (seconds) | Prior pit stops from `pit_stops.csv` |
| `constructor_pit_duration_std` | Pit Stop | `float` | Constructor pit stop consistency / variance | Prior pit stops from `pit_stops.csv` |

#DROP DATABSE IF EXISTS nba_shots;
#CREATE DATABASE nba_shots;

USE nba_shots;

# create table schema
DROP TABLE IF EXISTS all_shots; 
CREATE TABLE all_shots (
player_name varchar(156),
team_name varchar(156),
game_date date,
season int,
espn_player_id int, 
team_id int, 
espn_game_id int ,
period int ,
minutes_remaining int, 
seconds_remaining int, 
shot_made_flag int, 
action_type varchar(156),
shot_type varchar(156),
shot_distance int,
opponent varchar(156),
x int,
y int ,
dribbles int,
touch_time decimal(10,2),
defender_name varchar(156),
defender_distance decimal(10,2),
shot_clock decimal(10,2)
);

#load data into the file
LOAD DATA LOCAL INFILE "C:/Users/coreyjez/Documents/nba_draft/all_shots.csv"
INTO TABLE all_shots
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 LINES;



#create temp tables for 2 and 3 point fgs 
DROP TABLE IF EXISTS fg2; 
CREATE TEMPORARY TABLE fg2 AS (SELECT * FROM all_shots where shot_type = '2PT Field Goal');

DROP TABLE IF EXISTS fg3;
CREATE TEMPORARY TABLE fg3 AS (SELECT * FROM all_shots where shot_type = '3PT Field Goal');  

#create expected value by team for 2 point fgs 
DROP TABLE IF EXISTS fg2_ev; 
CREATE TEMPORARY TABLE fg2_ev AS (
			SELECT 
			fg2.team_name, 
			SUM(fg2.shot_made_flag) made_shots, 
			COUNT(1) shots_taken, 
			CAST(SUM(fg2.shot_made_flag) as decimal(10,2)) / CAST(COUNT(1) as decimal(10,2))*2 as ev_2
			FROM fg2 
			GROUP BY 1 

); 

DROP TABLE IF EXISTS fg3_ev; 
CREATE TEMPORARY TABLE fg3_ev AS (
				SELECT 
				fg3.team_name,
                fg3.shot_distance,
                round(fg3.defender_distance,0) defender_distance_whole,
                SUM(fg3.shot_made_flag) made_shots,
                COUNT(1) shots_taken,
                CAST(SUM(fg3.shot_made_flag) as decimal(10,2)) / CAST(COUNT(1) as decimal(10,2))*3 as ev_3
                FROM fg3
                
                GROUP BY 1,2,3			

) ;

#classify each 3PA whether or not its EV is > 2pt EV 
DROP TABLE IF EXISTS fg3_ev_class;
CREATE TEMPORARY TABLE fg3_ev_class AS (
				SELECT 
                fg3_ev.team_name,
                fg3_ev.shot_distance,
                fg3_ev.defender_distance_whole,
                ev_3,
                ev_2,
                CASE WHEN fg3_ev.ev_3 > fg2_ev.ev_2 then 1 else 0 end as q3pa 
				FROM fg3_ev
					JOIN fg2_ev on fg3_ev.team_name = fg2_ev.team_name

)  ;


#add column to raw fg3 dataset to get 
DROP TABLE IF EXISTS fg3_final;
CREATE TEMPORARY TABLE fg3_final AS (
SELECT 
f.*, c.q3pa, c.ev_3
FROM fg3 f
JOIN fg3_ev_class c on f.team_name = c.team_name and f.shot_distance = c.shot_distance and round(f.defender_distance,0) = c.defender_distance_whole
); 



SELECT
team_name,
quality_3pa, non_quality_3pa,
quality_3pa / (quality_3pa + non_quality_3pa) as q3pa_rate
FROM (
SELECT 
team_name, 
COUNT(CASE WHEN q3pa = 0 then 1 end) as non_quality_3pa,
COUNT(CASE WHEN q3pa = 1 then 1 end) as quality_3pa

FROM fg3_final

GROUP BY 1
ORDER BY 1
) a ;






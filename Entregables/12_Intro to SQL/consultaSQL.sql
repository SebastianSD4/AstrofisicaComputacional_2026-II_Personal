.open data/shows.db
.headers on 
.mode box 

-- 1. ¿Cuántas series tienen una calificación perfecta de 10.0? 

.print '1. ¿Cuántas series tienen una calificación perfecta de 10.0?'

SELECT * FROM ratings WHERE rating = 10;
SELECT COUNT(*) FROM ratings WHERE rating = 10;

-- 2. ¿Cuántos episodios tiene ’Black Mirror’?

.print ''
.print '2. ¿Cuántos episodios tiene ’Black Mirror’?'

SELECT episodes FROM shows where title = 'Black Mirror';

-- 3. ¿Cuántas series existen en el género de ciencia ficción?

.print ''
.print '3. ¿Cuántas series existen en el género de ciencia ficción?'

SELECT genre, COUNT(*) FROM genres WHERE genre = 'Sci-Fi';

-- 4. ¿Cuál es la serie mejor valorada del género de terror?

.print ''
.print '4. ¿Cuál es la serie mejor valorada del género de terror?'

SELECT title, 
(SELECT genre  FROM genres  WHERE show_id = shows.id AND genre = 'Terror') AS genre,
(SELECT rating FROM ratings WHERE show_id = shows.id) AS rating,
(SELECT votes  FROM ratings WHERE show_id = shows.id) AS votes
FROM shows 
WHERE id IN (
    SELECT show_id FROM ratings 
    WHERE show_id IN (SELECT show_id from genres WHERE genre = 'Horror')
)
ORDER BY rating DESC, votes DESC 
LIMIT 1;

-- 5. ¿Cuántas series de animación hay en la base de datos?

.print ''
.print '5. ¿Cuántas series de animación hay en la base de datos?'

SELECT genre, COUNT(*) FROM genres WHERE genre = 'Animation';

-- 6. ¿Cuáles fueron las 10 series de animación peor valoradas entre 2005 y 2010?

.print ''
.print '6. ¿Cuáles fueron las 10 series de animación peor valoradas entre 2005 y 2010?'

SELECT title, year,
(SELECT genre  FROM genres  WHERE show_id = shows.id AND genre = 'Animation') AS genre,
(SELECT rating FROM ratings WHERE show_id = shows.id) AS rating,
(SELECT votes  FROM ratings WHERE show_id = shows.id) AS votes
FROM shows
WHERE id IN (
    SELECT show_id from ratings 
    WHERE show_id in (SELECT show_id from genres WHERE genre = 'Animation')
    AND   show_id in (SELECT id from shows WHERE year BETWEEN 2005 AND 2010)
)
ORDER BY rating ASC, votes DESC
LIMIT 10; 



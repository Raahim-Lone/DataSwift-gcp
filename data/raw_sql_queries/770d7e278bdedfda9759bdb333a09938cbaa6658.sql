SELECT mi1.info, n.name, COUNT(*)
FROM title as t,
kind_type as kt,
movie_info as mi1,
info_type as it1,
cast_info as ci,
role_type as rt,
name as n,
info_type as it2,
person_info as pi
WHERE
t.id = ci.movie_id
AND t.id = mi1.movie_id
AND mi1.info_type_id = it1.id
AND t.kind_id = kt.id
AND ci.person_id = n.id
AND ci.movie_id = mi1.movie_id
AND ci.role_id = rt.id
AND n.id = pi.person_id
AND pi.info_type_id = it2.id
AND (it1.id IN ('2','5','6'))
AND (it2.id IN ('15'))
AND (mi1.info IN ('Canada:16+','Denmark:15','Germany:BPjM Restricted','New Zealand:R18','Peru:PT','Philippines:R-18','Portugal:M/16','Portugal:M/18','South Korea:All','Sweden:Btl'))
AND (n.name ILIKE '%th%')
AND (kt.kind IN ('episode','movie','video movie'))
AND (rt.role IN ('actress','costume designer','director','miscellaneous crew','producer'))
AND (t.production_year <= 2015)
AND (t.production_year >= 1975)
GROUP BY mi1.info, n.name;

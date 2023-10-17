SELECT_QUERY = '''SELECT building_id,
	   district_id,
	   count_floors_pre_eq,
	   count_floors_post_eq,
	   age_building,
	   plinth_area_sq_ft,
	   height_ft_pre_eq,
	   height_ft_post_eq,
	   land_surface_condition,
	   foundation_type,
	   roof_type,
	   ground_floor_type,
	   other_floor_type,
	   position,
	   plan_configuration,
	   condition_post_eq,
 		CASE
    		WHEN damage_grade = 'Grade 1' THEN 1
   			WHEN damage_grade = 'Grade 2' THEN 2
    		WHEN damage_grade = 'Grade 3' THEN 3
    		WHEN damage_grade = 'Grade 4' THEN 4
    		WHEN damage_grade = 'Grade 5' THEN 5
    		ELSE -1
  		END AS damage
    FROM Nepal_Earthquake'''
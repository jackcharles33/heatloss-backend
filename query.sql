WITH pivoted_questions AS (
  SELECT
    kf_job_reference,
    project_id,

    -- Property age: V1 uses era-band enums; V2 uses raw build year integer.
    -- We bucket V2 years into the same era bands _era() expects.
    COALESCE(
      property_age,
      CASE
        WHEN TRY_CAST(v2_property_build_year AS INTEGER) >= 2009 THEN 'POST_2008'
        WHEN TRY_CAST(v2_property_build_year AS INTEGER) BETWEEN 2000 AND 2008 THEN '2000_2008'
        WHEN TRY_CAST(v2_property_build_year AS INTEGER) BETWEEN 1960 AND 1999 THEN '1960_2000'
        WHEN TRY_CAST(v2_property_build_year AS INTEGER) < 1960 THEN 'PRE_1960'
        ELSE NULL
      END
    ) AS property_age,

    -- Roof type: COALESCE V1 room-level then V2 property-level
    COALESCE(roof_type, v2_roof_type) AS roof_type,

    -- Wall construction type: COALESCE V1 room-level then V2 property-level
    -- then V2 room-level alternate (max across rooms, per PIVOT semantics)
    COALESCE(
      walls_construction_type,
      v2_walls_construction_type,
      v2_room_alternate_wall_construction
    ) AS walls_construction_type,

    -- Windows glazing: COALESCE V1 room-level then V2 property-level
    COALESCE(windows_glazing, v2_windows_glazing) AS windows_glazing,

    -- Floor insulation (unchanged V1 logic, plus V2 additional_floor_insulation fallback)
    COALESCE(
      property_floor_insulation_type,
      CASE
        WHEN room_floor_insulation = 'true'  THEN 'HEAT_PUMP_SURVEY_PROPERTY_FLOOR_INSULATION_100'
        WHEN room_floor_insulation = 'false' THEN 'HEAT_PUMP_SURVEY_PROPERTY_FLOOR_INSULATION_0'
        ELSE room_floor_insulation
      END,
      v2_floor_insulation
    ) AS final_floor_insulation_type,

    -- Walls depth: FIX — BETWEEN_290_310 was previously left unmapped (10,653 rows)
    -- V2 fallback: EXTERNAL_WALL_THICKNESS_MM (560 rows, sparse but useful)
    CASE
      WHEN walls_depth IN (
        'HEAT_PUMP_SURVEY_MATERIAL_WALLS_DEPTH_MORE_THAN_290',
        'HEAT_PUMP_SURVEY_MATERIAL_WALLS_DEPTH_MORE_THAN_295',
        'HEAT_PUMP_SURVEY_MATERIAL_WALLS_DEPTH_MORE_THAN_310',
        'HEAT_PUMP_SURVEY_MATERIAL_WALLS_DEPTH_BETWEEN_290_310'
      ) THEN 'WALLS_DEPTH_GT_290'
      WHEN walls_depth IN (
        'HEAT_PUMP_SURVEY_MATERIAL_WALLS_DEPTH_LESS_THAN_290',
        'HEAT_PUMP_SURVEY_MATERIAL_WALLS_DEPTH_LESS_THAN_295'
      ) THEN 'WALLS_DEPTH_LT_290'
      -- V2 fallback: raw mm thickness from surveyor
      WHEN walls_depth IS NULL AND v2_walls_thickness_mm IS NOT NULL
        THEN CASE
          WHEN TRY_CAST(v2_walls_thickness_mm AS FLOAT) >= 290 THEN 'WALLS_DEPTH_GT_290'
          ELSE 'WALLS_DEPTH_LT_290'
        END
      ELSE walls_depth
    END AS final_walls_depth,

    -- Floor type: COALESCE V1 property-level then V2 property-level
    COALESCE(property_floor_type, v2_floor_type) AS property_floor_type,

    -- Roof insulation: prefer room-level, fall back to property-level, then V2
    COALESCE(
      room_roof_insulation_thickness,
      roof_insulation_thickness,
      v2_roof_insulation_thickness
    ) AS roof_insulation_thickness,

    -- Cavity fill status — COALESCE across all available signals.
    -- Preprocessing script will normalise these values to FILLED/UNFILLED.
    COALESCE(
      walls_insulation_type,
      room_walls_insulation_type,
      v2_cavity_wall_insulation,
      v2_cavity_insulation_prior_to_install,
      v2_retrospective_cavity_evidence,
      epc_cavity_insulation_impact
    ) AS walls_insulation,

    -- IWI/EWI thickness (mm) — directly informs solid wall U-value
    COALESCE(
      v2_iew_insulation_mm,
      v2_room_iew_insulation_mm
    ) AS wall_insulation_mm,

    -- EPC loft signal (secondary quality check on roof insulation)
    epc_loft_insulation_impact

  FROM (
    SELECT
      kf_job_reference,
      project_id,
      question,
      primary_response
    FROM
      octoenergy_data_prod_prod.consumer.wh_services_kfquestions
    WHERE
      job_type IN ('HEAT_PUMP_SURVEY', 'HEAT_PUMP_SURVEY_V2')
  )
  PIVOT (
    MAX(primary_response) FOR question IN (
      -- ── V1 fields ───────────────────────────────────────────────────────────
      'HEAT_PUMP_SURVEY_PROPERTY_AGE'                               AS property_age,
      'HEAT_PUMP_SURVEY_PROPERTY_GROUND_FLOOR_TYPE'                 AS property_floor_type,
      'HEAT_PUMP_SURVEY_PROPERTY_GROUND_FLOOR_INSULATION'           AS property_floor_insulation_type,
      'HEAT_PUMP_SURVEY_ROOM_FLOOR_INSULATION'                      AS room_floor_insulation,
      'HEAT_PUMP_SURVEY_ROOM_MATERIAL_ROOF_INSULATION_THICKNESS'    AS room_roof_insulation_thickness,
      'HEAT_PUMP_SURVEY_MATERIAL_ROOF_INSULATION_THICKNESS'         AS roof_insulation_thickness,
      'HEAT_PUMP_SURVEY_V2_ROOF_INSULATION_THICKNESS'               AS v2_roof_insulation_thickness,
      'HEAT_PUMP_SURVEY_ROOM_MATERIAL_ROOF_TYPE'                    AS roof_type,
      'HEAT_PUMP_SURVEY_ROOM_MATERIAL_WALLS_CONSTRUCTION_TYPE'      AS walls_construction_type,
      'HEAT_PUMP_SURVEY_ROOM_MATERIAL_WALLS_DEPTH'                  AS walls_depth,
      'HEAT_PUMP_SURVEY_ROOM_MATERIAL_WINDOWS_GLAZING'              AS windows_glazing,

      -- ── V1 wall insulation / cavity fill signals ─────────────────────────
      'HEAT_PUMP_SURVEY_MATERIAL_WALLS_INSULATION_TYPE'                          AS walls_insulation_type,
      'HEAT_PUMP_SURVEY_ROOM_MATERIAL_WALLS_INSULATION_TYPE'                     AS room_walls_insulation_type,
      'HEAT_PUMP_SURVEY_V2_CAVITY_WALL_INSULATION'                               AS v2_cavity_wall_insulation,
      'HEAT_PUMP_SURVEY_V2_CUSTOMER_CAVITY_WALL_INSULATION_PRIOR_TO_INSTALL'     AS v2_cavity_insulation_prior_to_install,
      'HEAT_PUMP_SURVEY_V2_RETROSPECTIVE_CAVITY_INSULATION_EVIDENCE'             AS v2_retrospective_cavity_evidence,

      -- ── V1/V2 IWI/EWI thickness for solid walls ──────────────────────────
      'HEAT_PUMP_SURVEY_V2_INTERNAL_OR_EXTERNAL_WALL_INSULATION_MM'              AS v2_iew_insulation_mm,
      'HEAT_PUMP_SURVEY_V2_ROOM_INTERNAL_OR_EXTERNAL_WALL_INSULATION_MM'         AS v2_room_iew_insulation_mm,

      -- ── V1/V2 EPC-derived signals ─────────────────────────────────────────
      'HEAT_PUMP_SURVEY_EPC_DATA_CAVITY_INSULATION_IMPACT'                       AS epc_cavity_insulation_impact,
      'HEAT_PUMP_SURVEY_EPC_DATA_LOFT_INSULATION_IMPACT'                         AS epc_loft_insulation_impact,

      -- ── NEW: V2 primary construction / building attributes ────────────────
      -- Property build year (integer, e.g. 1985) — bucketed to era band in SELECT above
      'HEAT_PUMP_SURVEY_V2_PROPERTY_BUILD_YEAR'                                  AS v2_property_build_year,

      -- Wall construction type (V2 equivalent of walls_construction_type)
      'HEAT_PUMP_SURVEY_V2_EXTERNAL_WALL_TYPE'                                   AS v2_walls_construction_type,

      -- Room-level wall construction override (V2 — alternate = surveyor override)
      'HEAT_PUMP_SURVEY_V2_ROOM_ALTERNATE_WALL_CONSTRUCTION'                     AS v2_room_alternate_wall_construction,

      -- Roof type (V2 property-level)
      'HEAT_PUMP_SURVEY_V2_ROOF_TYPE'                                            AS v2_roof_type,

      -- Window glazing type (V2 property-level)
      'HEAT_PUMP_SURVEY_V2_WINDOW_TYPE'                                          AS v2_windows_glazing,

      -- Floor type (V2 property-level)
      'HEAT_PUMP_SURVEY_V2_FLOOR_TYPE'                                           AS v2_floor_type,

      -- Floor insulation (V2 property-level)
      'HEAT_PUMP_SURVEY_V2_ADDITIONAL_FLOOR_INSULATION'                          AS v2_floor_insulation,

      -- Wall thickness mm (V2 property-level — used as walls_depth fallback)
      'HEAT_PUMP_SURVEY_V2_EXTERNAL_WALL_THICKNESS_MM'                           AS v2_walls_thickness_mm
    )
  )
)
SELECT
  t1.*,
  t2.ashp_survey_total_property_heatloss_w,
  t2.ashp_survey_total_floor_area_sqm
FROM pivoted_questions t1
LEFT JOIN octoenergy_data_prod_prod.coconut.dim_services_kfjobcharacteristics t2
  ON t1.project_id = t2.project_id
WHERE
  t2.ashp_survey_total_property_heatloss_w IS NOT NULL
  AND t1.walls_construction_type IS NOT NULL;
-- Note: LIMIT removed — full dataset is ~53k rows after adding HEAT_PUMP_SURVEY_V2

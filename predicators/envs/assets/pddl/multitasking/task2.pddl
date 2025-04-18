(define (problem room_manipulation_intermediate)
  (:domain room_domain)
  (:objects
    ;; Furniture
    table sofa shelf desk - furniture

    ;; Items
    book cup pen remote phone keys - item

    ;; Grippers
    left_gripper right_gripper - gripper

    ;; Locations
    loc0 loc1 loc2 loc3 loc4 loc5 - location
  )
  (:init
    ;; Furniture-to-location assignments
    (at table loc0)
    (at sofa loc1)
    (at shelf loc2)
    (at desk loc3)

    ;; Robot starts at loc5
    (robot_at loc5)

    ;; Items initially on furniture
    (item_at book table)
    (item_at cup sofa)
    (item_at pen desk)
    (item_at remote shelf)
    (item_at phone shelf)
    (item_at keys table)

    ;; Grippers are free
    (free left_gripper)
    (free right_gripper)

    ;; Reachability: all locations are reachable from each other
    (reachable loc0 loc1) (reachable loc1 loc0)
    (reachable loc0 loc2) (reachable loc2 loc0)
    (reachable loc0 loc3) (reachable loc3 loc0)
    (reachable loc0 loc4) (reachable loc4 loc0)
    (reachable loc0 loc5) (reachable loc5 loc0)
    (reachable loc1 loc2) (reachable loc2 loc1)
    (reachable loc1 loc3) (reachable loc3 loc1)
    (reachable loc1 loc4) (reachable loc4 loc1)
    (reachable loc1 loc5) (reachable loc5 loc1)
    (reachable loc2 loc3) (reachable loc3 loc2)
    (reachable loc2 loc4) (reachable loc4 loc2)
    (reachable loc2 loc5) (reachable loc5 loc2)
    (reachable loc3 loc4) (reachable loc4 loc3)
    (reachable loc3 loc5) (reachable loc5 loc3)
    (reachable loc4 loc5) (reachable loc5 loc4)
  )
  (:goal (and
    ;; Goal: more involved rearrangement for multiple pick/place tasks:
    ;; 1. Book from table -> desk
    (item_at book desk)
    ;; 2. Cup from sofa -> shelf
    (item_at cup shelf)
    ;; 3. Pen from desk -> sofa
    (item_at pen sofa)
    ;; 4. Remote from shelf -> table
    (item_at remote table)
    ;; 5. Phone from shelf -> desk
    (item_at phone desk)
    ;; 6. Keys from table -> sofa
    (item_at keys sofa)
  ))
)

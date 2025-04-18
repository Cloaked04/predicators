(define (domain room_domain)
    (:requirements :strips :typing)
    (:types 
        furniture   ; table, sofa, shelf
        item       ; items to manipulate
        gripper    ; robot grippers
        location   ; room locations
    )
    (:predicates
        (at ?f - furniture ?l - location)     ; furniture at location
        (item_at ?i - item ?f - furniture)    ; item on furniture
        (robot_at ?l - location)              ; robot position
        (free ?g - gripper)                   ; gripper is empty
        (holding ?i - item ?g - gripper)      ; gripper holding item
        (reachable ?l1 ?l2 - location)        ; robot can reach between locations
    )

    (:action move
        :parameters (?from ?to - location)
        :precondition (and 
            (robot_at ?from)
            (reachable ?from ?to)
        )
        :effect (and
            (not (robot_at ?from))
            (robot_at ?to)
        )
    )

    (:action pick
        :parameters (?item - item ?furniture - furniture ?loc - location ?gripper - gripper)
        :precondition (and
            (robot_at ?loc)
            (at ?furniture ?loc)
            (item_at ?item ?furniture)
            (free ?gripper)
        )
        :effect (and
            (not (item_at ?item ?furniture))
            (not (free ?gripper))
            (holding ?item ?gripper)
        )
    )

    (:action place
        :parameters (?item - item ?furniture - furniture ?loc - location ?gripper - gripper)
        :precondition (and
            (robot_at ?loc)
            (at ?furniture ?loc)
            (holding ?item ?gripper)
        )
        :effect (and
            (item_at ?item ?furniture)
            (free ?gripper)
            (not (holding ?item ?gripper))
        )
    )
)
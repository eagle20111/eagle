#coding=utf-8
import math

def WorldCoordToObjCoord(input_world_coord, obj_world_coord, obj_world_angle):
    """转换一组坐标为局部相对坐标,局部坐标系的x轴和heading一致

    Args:
        input_world_coord (tuple): _description_
        obj_world_coord (tuple): _description_
        obj_world_angle (tuple): _description_

    Returns:
        _type_: _description_
    """
    
    x_diff = input_world_coord[0] - obj_world_coord[0]
    y_diff = input_world_coord[1] - obj_world_coord[1]
    rho    = math.sqrt(x_diff * x_diff + y_diff * y_diff)#r长度
    theta  = math.atan2(y_diff, x_diff) - obj_world_angle#局部heading  当前heading减去之前的heading的差
    return (math.cos(theta) * rho,  math.sin(theta) * rho) #转换为以heading方向为局部坐标系的x轴

def WorldCoordToObjCoordNorth(input_world_coord, obj_world_coord, obj_world_angle):
    """转换一组坐标为局部相对坐标,heading方向为北

    Args:
        input_world_coord (tuple): _description_
        obj_world_coord (tuple): _description_
        obj_world_angle (tuple): _description_

    Returns:
        _type_: _description_
    """
    
    x_diff = input_world_coord[0] - obj_world_coord[0]
    y_diff = input_world_coord[1] - obj_world_coord[1]
    theta = math.pi/2 - obj_world_angle
    x = math.cos(theta)*x_diff - math.sin(theta)*y_diff
    y = math.sin(theta)*x_diff + math.cos(theta)*y_diff
    return (x,y)



# if __name__ == "__main__":
#     input_world_coord = [(5340.91455078, -9635.6796875), (5339.87109375, -9631.41699219), (-2,-2),(-2,2)]
#     obj_world_coord = [(5340.91455078, -9635.6796875) for _ in range(4)]
#     obj_world_angle = [-1.3163444059211]*4
#     for w_coord, obs_coord,obs_ang in zip(input_world_coord, obj_world_coord,obj_world_angle):
#         # print(WorldCoordToObjCoord(w_coord, obs_coord,obs_ang))
#         print(WorldCoordToObjCoordNorth(w_coord, obs_coord,obs_ang))
#         print("*"*10)



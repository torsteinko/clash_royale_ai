from roboflow import Roboflow
rf = Roboflow(api_key="Ko5Bjy2lz5wSX2Y52IBL")
project = rf.workspace("clashtest-jhyn1").project("merged_new_sameteam-h6wq1")
version = project.version(1)
dataset = version.download("yolov11")
                
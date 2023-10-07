generate:
	scale version
	scale signature generate bindings:latest
	scale signature export local/bindings:latest go host signature/host --manifest=false
	scale signature export local/bindings:latest rust guest signature/guest

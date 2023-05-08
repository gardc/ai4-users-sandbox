# AI4Users Sandbox Server
This is the repo for the AI4Users Sandbox. It's written in Python 3 with Flask for serving HTTP requests, and has Docker set up for easy deployment to any server.

## System architecture and dataflow
This diagram shows the dataflow of the three main functionalities of this system. <br/>
Module calls are mainly omitted. One exception is made, as that particular instance is essential in explaining its associated dataflow. <br/>
For further inspection, documentation is provided in their respective files. <br/> <br/>
![diagram](img/system_architecture.png?raw=true)

## Making sick leave estimate request
By looking at src/index.py, we can see the POST request to `/process_data` handles sick leave requests. Example body:
```
{
	"region": "Agder",
	"age": "16-19",
	"disorder": "pregnancy disorders",
	"gender": "female"
}
```

## Testing
cd to `src/tests/` and run `pytest` to perform tests.

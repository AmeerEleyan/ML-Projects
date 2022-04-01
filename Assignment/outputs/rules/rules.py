def findDecision(obj): #obj[0]: x, obj[1]: y
	
	# {"feature": "y", "instances": 400, "metric_value": 0.258, "depth": 1}
	if obj[1]<=-3.105319269587499:
		# {"feature": "x", "instances": 200, "metric_value": 0.0421, "depth": 2}
		if obj[0]>8.415644786075124:
			return 0
		elif obj[0]<=8.415644786075124:
			return 0
		else: return 0.25
	elif obj[1]>-3.105319269587499:
		# {"feature": "x", "instances": 200, "metric_value": 0.0413, "depth": 2}
		if obj[0]<=10.135222218528863:
			return 1
		elif obj[0]>10.135222218528863:
			return 0
		else: return 0.2857142857142857
	else: return 0.94

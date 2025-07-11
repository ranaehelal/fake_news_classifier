import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class FakeNewsScreen extends StatefulWidget {
  @override
  _FakeNewsScreenState createState() => _FakeNewsScreenState();
}

class _FakeNewsScreenState extends State<FakeNewsScreen> {
  TextEditingController _controller = TextEditingController();
  String? _result;
  bool _loading = false;

  Future<void> predictNews(String text) async {
    setState(() {
      _loading = true;
    });

    final url = Uri.parse('http://127.0.0.1:8000/predict');
    final response = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'text': text, 'model': "logistic"}),
    );

    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      setState(() {
        _result = data['prediction'];
        _loading = false;
      });
    } else {
      setState(() {
        _result = 'Error: ${response.statusCode}';
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        title: Text('📰 Fake News Detector'),
        centerTitle: true,
        elevation: 0,
        backgroundColor: Colors.deepPurple,
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Paste or write a news article below:',
              style: TextStyle(fontSize: 16, color: Colors.grey[800]),
            ),
            SizedBox(height: 12),
            Container(
              decoration: BoxDecoration(
                color: Colors.white,
                border: Border.all(color: Colors.deepPurple, width: 1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: TextField(
                controller: _controller,
                maxLines: 6,
                decoration: InputDecoration(
                  hintText: 'e.g. The president just signed a new law...',
                  contentPadding: EdgeInsets.all(12),
                  border: InputBorder.none,
                ),
              ),
            ),
            SizedBox(height: 20),
            _loading
                ? Center(child: CircularProgressIndicator())
                : ElevatedButton.icon(
              icon: Icon(Icons.search),
              label: Text('Analyze'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.deepPurple,
                padding: EdgeInsets.symmetric(vertical: 16),
                textStyle: TextStyle(fontSize: 16),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10)),
              ),
              onPressed: () => predictNews(_controller.text),
            ),
            SizedBox(height: 20),
            if (_result != null)
              Container(
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: _result == 'real' ? Colors.green[100] : Colors.red[100],
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                      color: _result == 'real' ? Colors.green : Colors.red,
                      width: 1),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      _result == 'real' ? Icons.check_circle : Icons.warning,
                      color: _result == 'real' ? Colors.green : Colors.red,
                    ),
                    SizedBox(width: 10),
                    Text(
                      'Prediction: ${_result!.toUpperCase()}',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: _result == 'real' ? Colors.green[800] : Colors.red[800],
                      ),
                    ),
                  ],
                ),
              )
          ],
        ),
      ),
    );
  }
}

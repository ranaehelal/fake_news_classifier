import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

import 'FakeNewsScreen.dart';

void main() {
  runApp(FakeNewsApp());
}

class FakeNewsApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fake News Detector',
      theme: ThemeData(primarySwatch: Colors.deepPurple),
      home: FakeNewsScreen(),
    );
  }
}

